"""Module implementing an orchestrator pattern using Azure OpenAI API.

This module demonstrates the orchestrator pattern for a blog post creation system that:
1. Plans the blog structure and sections
2. Writes individual sections
3. Reviews and polishes the final content

Features:
- Type-safe implementation using Python 3.11 type hints
- Pydantic models for structured outputs
- Azure OpenAI integration with configurable parameters
- Comprehensive error handling and logging
- Response metadata tracking
"""

import json
import logging
import os
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Initialize Azure OpenAI client
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

if not api_key or not endpoint:
    raise ValueError("Azure OpenAI credentials must be set in environment variables")

client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-10-21",
    azure_endpoint=endpoint
)
model = "gpt-4o"  # Azure OpenAI deployment name

# Reviewer prompt template
REVIEWER_PROMPT = """You are a skilled technical editor reviewing a blog post.
Topic: {topic}
Target Audience: {audience}

Review the following blog post sections and provide:
1. A cohesion score (0-1) indicating how well sections flow together, focusing on:
   - Logical progression of technical concepts
   - Consistent terminology and technical depth
   - Clear transitions between sections
   - References to previous sections when building on concepts
   - Connection between theoretical concepts and practical examples

2. Suggested edits for specific sections to improve:
   - Technical accuracy and clarity
   - Transitions between sections (e.g., "Building on our discussion of X...", "Having explored Y, let's examine...")
   - Consistent level of technical detail
   - Cross-references to related concepts
   - Code example integration with explanations
   - Practical applications of theoretical concepts

3. A complete, polished version that:
   - Maintains consistent technical terminology
   - Builds concepts progressively
   - Links theoretical and practical aspects
   - Provides clear technical transitions
   - References earlier concepts when introducing new ones

Blog Post:
{sections}"""


# Define prompts
ORCHESTRATOR_PROMPT = """
Analyze this blog topic and break it down into logical sections.

Topic: {topic}
Target Length: {target_length} words
Style: {style}

Return your response in this format:

# Analysis
Analyze the topic and explain how it should be structured.
Consider the narrative flow and how sections will work together.

# Target Audience
Define the target audience and their interests/needs.

# Sections
## Section 1
- Type: section_type
- Description: what this section should cover
- Style: writing style guidelines

[Additional sections as needed...]
"""

WORKER_PROMPT = """
Write a blog section based on:
Topic: {topic}
Section Type: {section_type}
Section Goal: {description}
Style Guide: {style_guide}

Return your response in this format:

# Content
[Your section content here, following the style guide]

# Key Points
- Main point 1
- Main point 2
[Additional points as needed...]
"""

REVIEWER_PROMPT = """
Review this blog post for cohesion and flow:

Topic: {topic}
Target Audience: {audience}

Sections:
{sections}

Provide a cohesion score between 0.0 and 1.0, suggested edits for each section if needed, and a final polished version of the complete post.

The cohesion score should reflect how well the sections flow together, with 1.0 being perfect cohesion.
For suggested edits, focus on improving transitions and maintaining consistent tone across sections.
The final version should incorporate your suggested improvements into a polished, cohesive blog post.
"""

class PlanResponse(BaseModel):
    """Response model for blog post planning."""
    topic_analysis: str = Field(description="Analysis of the blog topic")
    target_audience: str = Field(description="Intended audience for the blog")
    sections: List[Dict[str, str]] = Field(description="List of sections with type and description")


class SubTask(BaseModel):
    """Blog section task defined by orchestrator"""
    section_type: str = Field(description="Type of blog section to write")
    description: str = Field(description="What this section should cover")
    style_guide: str = Field(description="Writing style for this section")
    target_length: int = Field(description="Target word count for this section")


class OrchestratorPlan(BaseModel):
    """Orchestrator's blog structure and tasks"""
    topic_analysis: str = Field(description="Analysis of the blog topic")
    target_audience: str = Field(description="Intended audience for the blog")
    sections: List[SubTask] = Field(description="List of sections to write")


class SectionResponse(BaseModel):
    """Response model for section writing."""
    content: str = Field(description="Written content for the section")
    key_points: List[str] = Field(description="Main points covered")


class SectionContent(BaseModel):
    """Content written by a worker"""
    section_type: str = Field(description="Type of blog section")
    content: str = Field(description="Written content for the section")
    key_points: List[str] = Field(description="Main points covered")


class SuggestedEdits(BaseModel):
    """Suggested edits for a section"""
    section_name: str = Field(description="Name of the section")
    suggested_edit: str = Field(description="Suggested edit")


class ReviewResponse(BaseModel):
    """Response model for blog post review."""
    cohesion_score: float = Field(description="How well sections flow together (0-1)")
    suggested_edits: List[SuggestedEdits] = Field(description="Suggested edits by section")
    final_version: str = Field(description="Complete, polished blog post")


class ReviewFeedback(BaseModel):
    """Final review and suggestions"""
    cohesion_score: float = Field(description="How well sections flow together (0-1)")
    suggested_edits: List[SuggestedEdits] = Field(description="Suggested edits by section")
    final_version: str = Field(description="Complete, polished blog post")


class BlogOrchestrator:
    """Class implementing the orchestrator pattern for blog post creation.

    This class demonstrates the orchestrator pattern by:
    1. Planning: Break down blog post into sections
    2. Writing: Generate content for each section
    3. Reviewing: Evaluate and improve the final content

    Features:
    - Type-safe with Python 3.11 type hints
    - Pydantic models for structured outputs
    - Azure OpenAI integration
    - Comprehensive error handling and logging
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> None:
        """Initialize the BlogOrchestrator.

        Args:
            model: Azure OpenAI model deployment name
            temperature: Controls randomness in responses (0-1)
            max_tokens: Maximum tokens in response

        Raises:
            ValueError: If Azure OpenAI credentials are not set
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True
    )
    def plan_blog_post(
        self,
        topic: str,
        target_length: int,
        style: str
    ) -> OrchestratorPlan:
        """Break down a blog post into sections using LLM.

        Args:
            topic: Blog post topic
            target_length: Target word count
            style: Writing style guidelines

        Returns:
            OrchestratorPlan with sections and analysis

        Raises:
            ValueError: If inputs are invalid
            Exception: If the API call fails after all retries
        """
        if not topic or not topic.strip():
            raise ValueError("Topic cannot be empty")

        logger.info("Planning blog post structure")
        logger.debug("Topic: %s, Length: %d, Style: %s", topic, target_length, style)

        try:
            completion = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": ORCHESTRATOR_PROMPT.format(
                            topic=topic,
                            target_length=target_length,
                            style=style
                        )
                    }
                ],
                response_format=OrchestratorPlan,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract the parsed response
            plan = completion.choices[0].message.parsed
            
            # Add style and length to sections
            section_length = target_length // len(plan.sections)  # Even distribution
            for section in plan.sections:
                section.style_guide = style
                section.target_length = section_length
            
            return plan
            
        except Exception as e:
            logger.error("Error planning blog post: %s", str(e))
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True
    )
    def write_section(
        self,
        topic: str,
        section: SubTask
    ) -> SectionContent:
        """Write content for a blog section using LLM.

        Args:
            topic: Blog post topic
            section: Section details and requirements

        Returns:
            SectionContent with written content and key points

        Raises:
            ValueError: If inputs are invalid
            Exception: If the API call fails after all retries
        """
        if not topic or not topic.strip():
            raise ValueError("Topic cannot be empty")

        logger.info("Writing blog section: %s", section.section_type)
        logger.debug(
            "Section details - Type: %s, Description: %s",
            section.section_type, section.description
        )

        try:
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": WORKER_PROMPT.format(
                            topic=topic,
                            section_type=section.section_type,
                            description=section.description,
                            style_guide=section.style_guide
                        )
                    }
                ],
                response_format=SectionResponse,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            parsed = response.choices[0].message.parsed
            return SectionContent(
                section_type=section.section_type,
                content=parsed.content,
                key_points=parsed.key_points
            )
            
        except Exception as e:
            logger.error("Error writing section: %s", str(e))
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True
    )
    def review_post(
        self,
        topic: str,
        target_audience: str,
        sections: List[SectionContent]
    ) -> ReviewFeedback:
        """Review and polish the complete blog post using LLM.

        Args:
            topic: Blog post topic
            target_audience: Intended audience
            sections: List of written sections

        Returns:
            ReviewFeedback with cohesion score and suggested edits

        Raises:
            ValueError: If inputs are invalid
            Exception: If the API call fails after all retries
        """
        if not sections:
            raise ValueError("No sections provided for review")

        logger.info("Reviewing complete blog post")
        logger.debug("Number of sections: %d", len(sections))

        try:
            # Combine sections for review
            combined_sections = "\n\n".join(
                f"=== Section {i+1}. {section.section_type} ===\n{section.content}"
                for i, section in enumerate(sections)
            )

            completion = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": REVIEWER_PROMPT.format(
                            topic=topic,
                            audience=target_audience,
                            sections=combined_sections
                        )
                    }
                ],
                response_format=ReviewResponse,  # Use ReviewResponse for initial review
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract the parsed response
            feedback = completion.choices[0].message.parsed
            
            # Ensure cohesion score is between 0 and 1
            feedback.cohesion_score = max(0.0, min(1.0, feedback.cohesion_score))
            return feedback
            
        except Exception as e:
            logger.error("Error reviewing blog post: %s", str(e))
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True
    )
    def revise_section(
        self,
        topic: str,
        section_type: str,
        content: str,
        edit_suggestion: str,
        style: str
    ) -> SectionContent:
        """Revise a section based on review feedback.

        Args:
            topic: Blog post topic
            section_type: Type of section being revised
            content: Original content
            edit_suggestion: Suggested improvement
            style: Writing style guidelines

        Returns:
            Revised section content

        Raises:
            ValueError: If inputs are invalid
            Exception: If the API call fails after all retries
        """
        try:
            completion = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a skilled editor improving a blog section.
                            Original content: {content}
                            
                            Please revise this section following this suggestion:
                            {edit}
                            
                            Maintain the {style} style while making these improvements.
                            Return the revised content and key points covered.""".format(
                            content=content,
                            edit=edit_suggestion,
                            style=style
                        )
                    }
                ],
                response_format=SectionResponse,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            response = completion.choices[0].message.parsed
            return SectionContent(
                section_type=section_type,  # Preserve the original section type
                content=response.content,
                key_points=response.key_points
            )
            
        except Exception as e:
            logger.error("Error revising section: %s", str(e))
            raise

    def write_blog(
        self,
        topic: str,
        target_length: int,
        style: str
    ) -> Dict[str, any]:
        """Write a complete blog post using the orchestrator pattern.

        Args:
            topic: Blog post topic
            target_length: Target word count
            style: Writing style guidelines

        Returns:
            Dictionary containing plan, sections, review, and final content

        Raises:
            ValueError: If inputs are invalid
            Exception: If any step fails
        """
        # Step 1: Plan the blog post
        plan = self.plan_blog_post(topic, target_length, style)

        # Step 2: Write each section
        sections = []
        for section in plan.sections:
            content = self.write_section(topic, section)
            sections.append(content)

        # Step 3: Review and get suggestions
        review = self.review_post(topic, plan.target_audience, sections)

        # Step 4: Apply suggested edits
        if review.suggested_edits:
            revised_sections = []
            for i, section in enumerate(sections):
                # Find edit suggestion for this section
                section_number = f"Section {i+1}"
                edit = next(
                    (e for e in review.suggested_edits if section_number in e.section_name),
                    None
                )
                
                if edit:
                    # Revise section with suggested edit
                    revised = self.revise_section(
                        topic=topic,
                        section_type=plan.sections[i].section_type,
                        content=section.content,
                        edit_suggestion=edit.suggested_edit,
                        style=style
                    )
                    revised_sections.append(revised)
                else:
                    revised_sections.append(section)
            
            # Get final review of revised content
            # Combine revised sections for final review
            combined_revised = []
            for i, section in enumerate(revised_sections):
                # Use actual section index (1-based) for display
                section_num = i + 1
                section_header = f"=== Section {section_num}. {section.section_type} ===\n{section.content}"
                combined_revised.append(section_header)
            
            combined_revised = "\n\n".join(combined_revised)
            
            # Get final review to check cohesion improvements
            final_completion = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a skilled editor reviewing the revised blog post.
                            Topic: {topic}
                            Target Audience: {audience}
                            
                            Review the revised blog post and provide:
                            1. A cohesion score (0-1) indicating how well sections flow together
                            2. Verify that previous cohesion issues have been addressed
                            3. A complete, polished version that reads as one cohesive piece
                            
                            Blog Post:
                            {content}""".format(
                            topic=topic,
                            audience=plan.target_audience,
                            content=combined_revised
                        )
                    }
                ],
                response_format=ReviewResponse,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            final_review = ReviewFeedback(
                cohesion_score=final_completion.choices[0].message.parsed.cohesion_score,
                suggested_edits=[],  # No more edits needed
                final_version=final_completion.choices[0].message.parsed.final_version
            )
            
            return {
                "plan": plan,
                "original_sections": sections,
                "revised_sections": revised_sections,
                "initial_review": review,
                "final_review": final_review
            }

        return {
            "plan": plan,
            "sections": sections,
            "review": review
        }


def main() -> None:
    """Example usage of the orchestrator pattern for creating a technical blog post."""
    # Initialize orchestrator with lower temperature for consistency
    orchestrator = BlogOrchestrator(temperature=0.1)

    # Example: Technical blog post
    topic = "The impact of AI on software development"
    result = orchestrator.write_blog(
        topic=topic,
        target_length=1200,
        style="technical but accessible"
    )

    if "final_review" in result:
        print("\n=== Initial Draft ===\n")
        print(result["initial_review"].final_version)
        print(f"\nInitial Cohesion Score: {result['initial_review'].cohesion_score}")

        print("\n=== Suggested Edits ===\n")
        for edit in result["initial_review"].suggested_edits:
            print(f"Section: {edit.section_name}")
            print(f"Suggested Edit: {edit.suggested_edit}")

        print("\n=== Revised Version ===\n")
        print(result["final_review"].final_version)
        print(f"\nFinal Cohesion Score: {result['final_review'].cohesion_score}")
    else:
        print("\n=== Blog Post ===\n")
        print(result["review"].final_version)
        print(f"\nCohesion Score: {result['review'].cohesion_score}")
        
        if result["review"].suggested_edits:
            print("\n=== Suggested Edits ===\n")
            for edit in result["review"].suggested_edits:
                print(f"Section: {edit.section_name}")
                print(f"Suggested Edit: {edit.suggested_edit}")


if __name__ == "__main__":
    main()
