"""
LLM Backend for Customer Churn Explanation using NVIDIA API
"""

from openai import OpenAI
import re


def generate_churn_explanation(customer_data, system_prompt):
    """
    Generate explanation for churn prediction using NVIDIA LLM.

    Args:
        customer_data (dict): Dictionary containing customer parameters and prediction
            {
                'credit_score': int,
                'geography': str,
                'gender': str,
                'age': int,
                'tenure': int,
                'balance': float,
                'num_of_products': int,
                'has_cr_card': str ('Yes'/'No'),
                'is_active_member': str ('Yes'/'No'),
                'estimated_salary': float,
                'prediction': str ('WILL EXIT' / 'WILL NOT EXIT'),
                'probability': str (percentage),
                'confidence': str (percentage)
            }
        system_prompt (str): System instruction for the LLM

    Returns:
        str: LLM-generated explanation or None if error
    """
    try:
        # Initialize NVIDIA-compatible OpenAI client
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="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"
        )

        # Construct user input with customer data
        user_input = f"""
Customer Profile:
- Credit Score: {customer_data['credit_score']}
- Geography: {customer_data['geography']}
- Gender: {customer_data['gender']}
- Age: {customer_data['age']} years
- Tenure: {customer_data['tenure']} years
- Account Balance: ${customer_data['balance']:,.2f}
- Number of Products: {customer_data['num_of_products']}
- Has Credit Card: {customer_data['has_cr_card']}
- Is Active Member: {customer_data['is_active_member']}
- Estimated Salary: ${customer_data['estimated_salary']:,.2f}

Prediction Result:
- Outcome: {customer_data['prediction']}
- Probability: {customer_data['probability']}
- Confidence: {customer_data['confidence']}

Please provide a detailed explanation of this churn prediction.
"""

        # Send request to NVIDIA LLM
        completion = client.chat.completions.create(
            model="meta/llama3-70b-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            top_p=1,
            max_tokens=1024,
            stream=False
        )

        # Extract response text
        response_text = completion.choices[0].message.content

        # Clean up code fences or formatting
        clean_text = re.sub(r"^```(?:\w+)?", "", response_text.strip(), flags=re.MULTILINE)
        clean_text = re.sub(r"```$", "", clean_text.strip())

        return clean_text

    except Exception as e:
        print(f"LLM Error: {e}")
        return f"Error generating explanation: {str(e)}"


# Example usage for testing
if __name__ == "__main__":
    # Sample customer data
    sample_data = {
        'credit_score': 600,
        'geography': 'France',
        'gender': 'Male',
        'age': 40,
        'tenure': 3,
        'balance': 60000.00,
        'num_of_products': 2,
        'has_cr_card': 'Yes',
        'is_active_member': 'Yes',
        'estimated_salary': 50000.00,
        'prediction': 'WILL EXIT',
        'probability': '75.50%',
        'confidence': '51.00%'
    }

    sample_prompt = "You are a banking analytics expert. Explain churn predictions clearly."

    result = generate_churn_explanation(sample_data, sample_prompt)
    print(result)