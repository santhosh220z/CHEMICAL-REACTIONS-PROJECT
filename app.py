"""Chemical Reaction Predictor - Flask web app.

Predicts which chemical reaction occurs given reactant names, their masses (g),
and temperature (C), then computes limiting reagent and product amounts.
"""

from flask import Flask, render_template, jsonify, request

from core.engine import predict, EngineError, NoReactionError
from core.catalog import all_entries
from core.templates import reaction_types

app = Flask(__name__)


def _serialize(result):
    """Convert engine result to a JSON-safe dict."""
    return {
        'matched': result['matched'],
        'reaction_name': result['reaction_name'],
        'reaction_type': result['reaction_type'],
        'equation': result['equation'],
        'equation_display': result['equation_display'],
        'notes': result['notes'],
        'limiting_reagent': result['limiting_reagent_name'],
        'products': result['products'],
        'leftover': result['leftover'],
        'description': result.get('description'),
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chemicals')
def api_chemicals():
    """Return the catalog so the frontend can offer a picker."""
    return jsonify([{
        'name': e['name'],
        'formula': e['formula'],
        'category': e.get('category'),
        'phase': e.get('phase'),
    } for e in all_entries()])


@app.route('/api/reaction-types')
def api_reaction_types():
    return jsonify(reaction_types())


@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json(silent=True) or {}
    reactants = data.get('reactants') or []
    temperature = data.get('temperature')

    if temperature is None or temperature == '':
        return jsonify({'error': 'Provide a temperature in deg C.'}), 400
    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        return jsonify({'error': 'Temperature must be a number.'}), 400

    cleaned = []
    for r in reactants:
        name = (r.get('name') or '').strip()
        mass = r.get('mass_g')
        if not name:
            return jsonify({'error': 'Every reactant needs a name.'}), 400
        try:
            mass = float(mass)
        except (TypeError, ValueError):
            return jsonify({'error': f"Mass for '{name}' must be a number (grams)."}), 400
        if mass <= 0:
            return jsonify({'error': f"Mass for '{name}' must be positive."}), 400
        cleaned.append({'name': name, 'mass_g': mass})

    if not cleaned:
        return jsonify({'error': 'Provide at least one reactant.'}), 400

    try:
        result = predict(cleaned, temperature=temperature)
    except NoReactionError as e:
        return jsonify({'matched': False, 'error': str(e)}), 200
    except EngineError as e:
        return jsonify({'matched': False, 'error': str(e)}), 200
    except Exception as e:  # unexpected failure (e.g. network in resolver)
        app.logger.exception('prediction failed')
        return jsonify({'matched': False, 'error': f'Internal error: {e}'}), 500

    return jsonify(_serialize(result))


if __name__ == '__main__':
    app.run(debug=True)