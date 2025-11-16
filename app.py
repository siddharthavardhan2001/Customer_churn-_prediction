"""
Flask Web Application for Customer Churn Prediction
"""

from flask import Flask, render_template, request, jsonify

# Import prediction function from predict.py
from predict import predict_churn
# Import LLM backend
from llm_backend import generate_churn_explanation
# Import system prompt
from system_prompt import SYSTEM_PROMPT

app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Get form data
        credit_score = int(request.form.get('credit_score'))
        geography = request.form.get('geography')
        gender = request.form.get('gender')
        age = int(request.form.get('age'))
        tenure = int(request.form.get('tenure'))
        balance = float(request.form.get('balance'))
        num_of_products = int(request.form.get('num_of_products'))
        has_cr_card = int(request.form.get('has_cr_card'))
        is_active_member = int(request.form.get('is_active_member'))
        estimated_salary = float(request.form.get('estimated_salary'))

        # Make prediction
        result = predict_churn(
            credit_score=credit_score,
            geography=geography,
            gender=gender,
            age=age,
            tenure=tenure,
            balance=balance,
            num_of_products=num_of_products,
            has_cr_card=has_cr_card,
            is_active_member=is_active_member,
            estimated_salary=estimated_salary
        )

        # Prepare customer data for LLM
        customer_data = {
            'credit_score': credit_score,
            'geography': geography,
            'gender': gender,
            'age': age,
            'tenure': tenure,
            'balance': balance,
            'num_of_products': num_of_products,
            'has_cr_card': 'Yes' if has_cr_card == 1 else 'No',
            'is_active_member': 'Yes' if is_active_member == 1 else 'No',
            'estimated_salary': estimated_salary,
            'prediction': 'WILL EXIT' if result['will_churn'] else 'WILL NOT EXIT',
            'probability': f"{result['probability']*100:.2f}%",
            'confidence': f"{result['confidence']*100:.2f}%"
        }

        # Generate AI explanation
        ai_explanation = generate_churn_explanation(customer_data, SYSTEM_PROMPT)

        # Format response
        response = {
            'success': True,
            'will_churn': 'YES' if result['will_churn'] else 'NO',
            'probability': f"{result['probability']*100:.2f}%",
            'confidence': f"{result['confidence']*100:.2f}%",
            'ai_explanation': ai_explanation if ai_explanation else 'AI explanation unavailable'
        }

        return jsonify(response)

    except FileNotFoundError as e:
        return jsonify({
            'success': False,
            'error': 'Model not found. Please run main.py first to train the model.'
        })
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid input: {str(e)}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)