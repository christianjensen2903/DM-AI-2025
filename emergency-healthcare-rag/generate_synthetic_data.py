import json
import os
import random
import argparse
from pathlib import Path
from openai import OpenAI
import dotenv

dotenv.load_dotenv()


class SyntheticDataGenerator:
    def __init__(self, base_path: str = "data"):
        """Initialize the synthetic data generator.

        Args:
            api_key: OpenAI API key
            base_path: Base path to data directory
        """
        self.client = OpenAI()
        self.base_path = Path(base_path)
        self.topics_file = self.base_path / "topics.json"
        self.cleaned_topics_dir = self.base_path / "cleaned_topics"
        self.synthetic_dir = self.base_path / "synthetic"

        # Create synthetic directory structure
        self.synthetic_statements_dir = self.synthetic_dir / "statements"
        self.synthetic_answers_dir = self.synthetic_dir / "answers"

        self._load_topics()
        self._create_directories()

    def _load_topics(self) -> None:
        """Load topics from topics.json."""
        with open(self.topics_file, "r") as f:
            self.topics = json.load(f)
        self.topic_names = list(self.topics.keys())
        print(f"Loaded {len(self.topic_names)} topics")

    def _create_directories(self) -> None:
        """Create synthetic data directories if they don't exist."""
        self.synthetic_statements_dir.mkdir(parents=True, exist_ok=True)
        self.synthetic_answers_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directories: {self.synthetic_dir}")

    def _get_existing_statement_count(self) -> int:
        """Get the count of existing synthetic statements to continue numbering."""
        existing_files = list(self.synthetic_statements_dir.glob("statement_*.txt"))
        if not existing_files:
            return 0

        # Extract numbers from filenames and find the highest
        numbers = []
        for file in existing_files:
            try:
                num = int(file.stem.split("_")[1])
                numbers.append(num)
            except (IndexError, ValueError):
                continue

        return max(numbers) + 1 if numbers else 0

    def _read_topic_articles(self, topic_name: str) -> str:
        """Read all articles for a given topic.

        Args:
            topic_name: Name of the topic

        Returns:
            Combined content of all articles for the topic
        """
        topic_dir = self.cleaned_topics_dir / topic_name
        if not topic_dir.exists():
            print(f"Warning: Topic directory {topic_dir} does not exist")
            return ""

        articles = []
        for md_file in topic_dir.glob("*.md"):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    articles.append(f"=== {md_file.name} ===\n{content}")
            except Exception as e:
                print(f"Error reading {md_file}: {e}")

        return "\n\n".join(articles)

    def _generate_statement_prompt(
        self, topic_name: str, articles: str, should_be_true: bool
    ) -> str:
        """Generate the prompt for OpenAI to create a medical statement.

        Args:
            topic_name: Name of the medical topic
            articles: Combined article content
            should_be_true: Whether the statement should be factually correct

        Returns:
            Formatted prompt for OpenAI
        """
        truthfulness = (
            "factually correct and true"
            if should_be_true
            else "factually incorrect or false"
        )

        prompt = f"""You are a medical expert tasked with creating educational statements about healthcare topics.

Topic: {topic_name}

Based on the following medical articles, create a single, clear medical statement that is {truthfulness}.

Medical Articles:
{articles}

Requirements:
1. Create exactly ONE statement (1-2 sentences maximum)
2. The statement must be about {topic_name}
3. The statement must be {truthfulness}
4. The statement must be specific enough to be checked directly against the reference content
5. If creating a false statement, make it subtly incorrect but plausible-sounding
6. Use professional medical language
7. Do not include explanations or additional text
8. Return only the statement itself
9. Do NOT quote the article verbatim. Paraphrase.
10. Rotate between anatomy, epidemiology, pathophysiology, diagnosis, and management so the statements don't all hit the same angle.

Here are some examples of statements (on various topics):
- Coronary heart disease affects approximately 15.5 million people in the United States, with the American Heart Association estimating that a person experiences a heart attack every 41 seconds.
- Neonatal testicular torsion typically occurs as extravaginal torsion because the tunica vaginalis has not adhered to the gubernaculum, allowing both the tunica vaginalis and spermatic cord to twist together.
- Patients with acute myocardial infarction show a biphasic BNP response, with smaller infarcts peaking at 20 hours and larger infarcts showing an additional peak at 10 days post-admission.
- In euglycemic diabetic ketoacidosis management, dextrose 5% should be added to initial fluid resuscitation since glucose levels are less than 250 mg/dL, and insulin infusion rates should be started at 0.05-0.1 U/kg/hr rather than the higher doses used in hyperglycemic DKA.
- In inferior wall myocardial infarction, the mortality rate is less than 10%, but when complicated by right ventricular involvement, mortality increases to more than 25%.
- Cardiac syncope carries a one-year mortality rate of 30% and is estimated to be the cause of syncope in 15% of all syncopal events, with ventricular tachycardia alone being responsible for 11% of syncopal episodes.
- Idarucizumab is the recommended reversal agent for both dabigatran and rivaroxaban-associated bleeding, while andexanet alfa is specifically used for apixaban reversal only.
- Patients with sickle cell anemia, thalassemia, and those receiving blood transfusions require HbA1c testing every 2 months instead of the standard 3-month interval due to altered red blood cell turnover affecting test reliability.


Statement:"""

        return prompt

    def _generate_statement(
        self, topic_name: str, articles: str, should_be_true: bool
    ) -> str:
        """Generate a medical statement using OpenAI.

        Args:
            topic_name: Name of the medical topic
            articles: Combined article content
            should_be_true: Whether statement should be true or false

        Returns:
            Generated medical statement or empty string if generation fails
        """
        prompt = self._generate_statement_prompt(topic_name, articles, should_be_true)

        try:
            response = self.client.chat.completions.create(
                # model="gpt-4o",
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical expert who creates educational statements for training purposes.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.7,
            )

            statement = response.choices[0].message.content.strip()

            # Clean up the statement - remove any extra formatting
            if statement.startswith("Statement:"):
                statement = statement[10:].strip()

            return statement

        except Exception as e:
            print(f"Error generating statement: {e}")
            return ""

    def _save_statement_and_answer(
        self, statement: str, topic_name: str, is_true: bool, statement_id: int
    ) -> None:
        """Save the generated statement and corresponding answer.

        Args:
            statement: The generated medical statement
            topic_name: Name of the topic
            is_true: Whether the statement is true or false
            statement_id: Numeric ID for the statement
        """
        # Save statement
        statement_filename = f"statement_{statement_id:04d}.txt"
        statement_path = self.synthetic_statements_dir / statement_filename

        with open(statement_path, "w", encoding="utf-8") as f:
            f.write(statement)

        # Save answer
        answer_filename = f"statement_{statement_id:04d}.json"
        answer_path = self.synthetic_answers_dir / answer_filename

        answer_data = {
            "statement_is_true": 1 if is_true else 0,
            "statement_topic": self.topics[topic_name],
        }

        with open(answer_path, "w", encoding="utf-8") as f:
            json.dump(answer_data, f, indent=2)

        print(
            f"Saved statement {statement_id:04d}: {topic_name} ({'TRUE' if is_true else 'FALSE'})"
        )

    def generate_synthetic_data(self, n: int) -> None:
        """Generate n synthetic medical statements.

        Args:
            n: Number of statements to generate
        """
        print(f"Generating {n} synthetic statements...")

        starting_id = self._get_existing_statement_count()
        print(f"Starting from statement ID: {starting_id}")

        successful_generations = 0
        attempts = 0

        while (
            successful_generations < n and attempts < n * 2
        ):  # Max 2x attempts to handle failures
            attempts += 1

            # Randomly select topic
            topic_name = random.choice(self.topic_names)

            # Read articles for this topic
            articles = self._read_topic_articles(topic_name)
            if not articles:
                print(f"No articles found for topic: {topic_name}, skipping...")
                continue

            # Flip coin for truthfulness
            should_be_true = random.choice([True, False])

            # Generate statement
            statement = self._generate_statement(topic_name, articles, should_be_true)
            if not statement:
                print(f"Failed to generate statement for {topic_name}, retrying...")
                continue

            # Save statement and answer
            statement_id = starting_id + successful_generations
            self._save_statement_and_answer(
                statement, topic_name, should_be_true, statement_id
            )

            successful_generations += 1

        print(f"\nGeneration complete!")
        print(f"Successfully generated: {successful_generations}/{n} statements")
        print(f"Total attempts: {attempts}")


def main():
    """Main function to run the synthetic data generator."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic healthcare statements"
    )
    parser.add_argument("n", type=int, help="Number of statements to generate")

    args = parser.parse_args()

    # Validate inputs
    if args.n <= 0:
        print("Error: Number of statements must be positive")
        return

    # Initialize generator
    try:
        generator = SyntheticDataGenerator()
    except Exception as e:
        print(f"Error initializing generator: {e}")
        return

    # Generate synthetic data
    try:
        generator.generate_synthetic_data(args.n)
    except Exception as e:
        print(f"Error during generation: {e}")
        return


if __name__ == "__main__":
    main()
