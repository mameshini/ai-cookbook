"""RAG Evaluation Module using Amazon Bedrock Knowledge Bases and RAGAS framework.

This module provides functionality for evaluating Q&A applications using Amazon Bedrock
Knowledge Bases along with LangChain and RAGAS for response evaluation.
We will query the knowledge base to get the desired number of document chunks based on similarity search, prompt the query using Anthropic Claude, and then evaluate the responses effectively using evaluation metrics, such as faithfulness, answer_relevancy, context_recall, context_precision, context_entity_recall, answer_similarity, answer_correctness, harmfulness, maliciousness, coherence, correctness and conciseness.
"""

import os
from typing import List
import asyncio
import boto3
import datetime
import csv
from tqdm import tqdm
import pandas as pd
from botocore.config import Config
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_aws.embeddings.bedrock import BedrockEmbeddings
from langchain_aws.retrievers.bedrock import AmazonKnowledgeBasesRetriever
from langchain.chains import RetrievalQA
from ragas import SingleTurnSample 
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall
)
from ragas.metrics._factual_correctness import FactualCorrectness

class RagEval:
    """Knowledge Base RAG Evaluation class for assessing Q&A performance."""

    def __init__(
        self,
        kb_id: str,
        region_name: str = "us-west-2",
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
        eval_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        num_results: int = 5
    ):
        """Initialize the RAG evaluator.

        Args:
            kb_id: Knowledge Base ID to query against
            region_name: AWS region name
            model_id: Bedrock model ID for generating responses
            eval_model_id: Bedrock model ID for evaluation
            num_results: Number of results to retrieve from knowledge base
        """
        self.kb_id = kb_id
        self.region_name = region_name
        self.num_results = num_results

        # Initialize AWS clients with explicit region
        bedrock_config = Config(
            region_name=region_name,
            connect_timeout=120,
            read_timeout=120,
            retries={'max_attempts': 0}
        )
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=region_name,
            config=bedrock_config
        )
        self.bedrock_agent_client = boto3.client(
            "bedrock-agent-runtime",
            region_name=region_name,
            config=bedrock_config
        )

        # Initialize LangChain models with region
        self.llm = ChatBedrock(
            model_id=model_id,
            client=self.bedrock_client,
            region_name=region_name
        )
        self.eval_llm = ChatBedrock(
            model_id=eval_model_id,
            client=self.bedrock_client,
            region_name=region_name
        )
        self.embeddings_llm = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            client=self.bedrock_client,
            region_name=region_name
        )

        # Initialize retriever with explicit client
        self.retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=kb_id,
            client=self.bedrock_agent_client,
            region_name=region_name,
            retrieval_config={"vectorSearchConfiguration": {"numberOfResults": num_results}}
        )

        # Initialize QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True
        )

        # Test KB is working
        query = "Provide a list of few risks for Octank financial in numbered list without description."

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, retriever=self.retriever, return_source_documents=True
        )
        response = qa_chain.invoke(query)
        print(response["result"])


    async def evaluate_responses(
        self,
        questions: List[str],
        ground_truths: List[str]
    ) -> pd.DataFrame:
        """Evaluate responses using RAGAS metrics.

        Args:
            questions: List of questions to evaluate
            ground_truths: List of ground truth answers

        Returns:
            DataFrame containing evaluation results
        """
        answers = []
        contexts = []

        for query in questions:
            answers.append(self.qa_chain.invoke(query)["result"])
            contexts.append([docs.page_content for docs in self.retriever.invoke(query)])

        # Create dataset df
        dataset_df = pd.DataFrame(list(zip(questions, ground_truths, answers)),
                        columns=['questions', 'ground_truths', 'answers'])

        # Evaluation using RAGAS
        # Use the `evaluate()` function and simply pass in the relevant metrics and the prepared dataset.

        metrics = { 'context_precision': LLMContextPrecisionWithReference,
                    'context_recall': LLMContextRecall, 
                    'factual_correctness': FactualCorrectness
                }

        # Model Configuration
        llm_for_evaluation = LangchainLLMWrapper(self.eval_llm)
        bedrock_embeddings = LangchainEmbeddingsWrapper(self.embeddings_llm)

        # Create an empty DataFrame
        df_retrieve = pd.DataFrame(columns=list(metrics.keys()))

        score_list = []
        # Outer progress bar for samples
        with tqdm(total=len(questions), desc="Evaluating samples") as pbar_samples:
            for idx, (question, ground_truth, context, answer) in enumerate(zip(questions, ground_truths, contexts, answers)):
                score_dict = {}
                # Inner progress bar for metrics
                with tqdm(total=len(metrics), desc=f"Sample {idx+1} metrics", leave=False) as pbar_metrics:
                    for k, v in metrics.items():
                        scorer = v(llm=llm_for_evaluation)
                        sample = SingleTurnSample(
                                    user_input=question,
                                    response=answer,
                                    reference=ground_truth,
                                    retrieved_contexts=context, 
                                    )
                        score_result = await scorer.single_turn_ascore(sample)
                        score_dict[k] = score_result
                        pbar_metrics.update(1)
                score_list.append(score_dict)
                pbar_samples.update(1)

        # convert score_list to df
        df_metrics = pd.DataFrame(score_list)

        # merge metric df with dataset df
        results = pd.merge(dataset_df, df_metrics, left_index=True, right_index=True)

        # Save results to CSV file
        current_time = datetime.datetime.now().strftime("%m%d%y %H:%M")
        eval_dir = os.path.join(os.path.dirname(__file__), 'rag_evals')
        os.makedirs(eval_dir, exist_ok=True)
        csv_filename = os.path.join(eval_dir, f"rag_eval_{current_time}.csv")
        results.to_csv(csv_filename, index=False, escapechar='\\', quoting=csv.QUOTE_ALL)
        print(f"Results saved to {csv_filename}")

        return results


async def main() -> None:
    """Main function to demonstrate the RAG evaluation pipeline."""
    # Get Knowledge Base ID
    kb_id = os.getenv('KNOWLEDGE_BASE_ID')
    if not kb_id:
        kb_id = "Y5UHFTTWAT"  # Default KB ID for testing
        print(f"Using default Knowledge Base ID: {kb_id}")

    # Initialize evaluator
    evaluator = RagEval(kb_id=kb_id)

    # Example questions and ground truths
    questions = [
        "What was the primary reason for the increase in net cash provided by operating activities for Octank Financial in 2021?",
        "In which year did Octank Financial have the highest net cash used in investing activities, and what was the primary reason for this?",
        "What was the primary source of cash inflows from financing activities for Octank Financial in 2021?",
        "Calculate the year-over-year percentage change in cash and cash equivalents for Octank Financial from 2020 to 2021.",
        "Based on the information provided, what can you infer about Octank Financial's overall financial health and growth prospects?"
    ]
    
    ground_truths = [
        "The increase in net cash provided by operating activities was primarily due to an increase in net income and favorable changes in operating assets and liabilities.",
        "Octank Financial had the highest net cash used in investing activities in 2021, at $360 million, compared to $290 million in 2020 and $240 million in 2019. The primary reason for this was an increase in purchases of property, plant, and equipment and marketable securities.",
        "The primary source of cash inflows from financing activities for Octank Financial in 2021 was an increase in proceeds from the issuance of common stock and long-term debt.",
        "To calculate the year-over-year percentage change in cash and cash equivalents from 2020 to 2021: 2020 cash and cash equivalents: $350 million, 2021 cash and cash equivalents: $480 million. The percentage change is ($480M - $350M)/$350M = 37.1% increase.",
        "Based on the financial data, Octank Financial shows strong growth prospects with increasing operating cash flows, strategic investments in property and equipment, successful capital raising through stock and debt issuance, and a healthy increase in cash reserves. These indicators suggest solid operational performance and effective capital management."
    ]

    # Run evaluation
    print("\nStarting RAG evaluation...")
    results = await evaluator.evaluate_responses(questions, ground_truths)

    # Display results
    print("\nEvaluation Results:")
    print(results.to_string())


if __name__ == "__main__":
    asyncio.run(main())
