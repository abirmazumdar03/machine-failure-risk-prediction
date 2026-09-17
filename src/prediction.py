import pandas as pd

from src.data_processing import add_engineered_features


def interpret_risk(probability, threshold=0.5):
    """Translate a model probability into a simple maintenance-oriented label."""
    if probability >= max(threshold, 0.7):
        return "High Failure Risk"
    if probability >= threshold:
        return "Elevated Failure Risk"
    return "Low Failure Risk"


def _important_characteristics(machine_data, reference_data=None):
    """Describe notable input characteristics without claiming a diagnosis."""
    notes = []

    if reference_data is None or reference_data.empty:
        return notes

    numeric_columns = reference_data.select_dtypes(include="number").columns

    for column in numeric_columns:
        if column not in machine_data:
            continue

        value = machine_data[column]
        low = reference_data[column].quantile(0.25)
        high = reference_data[column].quantile(0.75)

        if value > high:
            notes.append(f"{column} is above the training-data upper quartile")
        elif value < low:
            notes.append(f"{column} is below the training-data lower quartile")

    return notes[:3]


def predict_machine_failure(
    model,
    machine_characteristics,
    threshold=0.5,
    reference_data=None,
):
    """Predict failure risk for one machine from its operating characteristics."""
    machine_df = pd.DataFrame([machine_characteristics])
    machine_df = add_engineered_features(machine_df)

    probability = float(model.predict_proba(machine_df)[:, 1][0])
    predicted_class = int(probability >= threshold)
    risk_label = interpret_risk(probability, threshold=threshold)
    contributing_characteristics = _important_characteristics(
        machine_df.iloc[0].to_dict(),
        reference_data=reference_data,
    )

    if predicted_class == 1:
        recommendation = "High predicted failure risk. Further inspection recommended."
    else:
        recommendation = "Predicted failure risk is low. Continue routine monitoring."

    return {
        "failure_probability": probability,
        "predicted_class": predicted_class,
        "risk_interpretation": risk_label,
        "notable_characteristics": contributing_characteristics,
        "recommended_next_step": recommendation,
    }
