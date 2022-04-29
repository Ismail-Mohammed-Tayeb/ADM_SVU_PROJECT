def create_age_groups(age):
    if 28 <= age <= 40:
        return "young"
    if 40 < age <= 55:
        return "man"
    if 55 < age <= 66:
        return "old"
    else:
        return "Unknown"


def create_heart_rate_groups(max_heart_rate):
    if max_heart_rate <= 114:
        return "very_low"
    if 114 < max_heart_rate <= 133:
        return "low"
    if 133 < max_heart_rate <= 152:
        return "moderate"
    if 152 < max_heart_rate <= 171:
        return "high"
    if 171 < max_heart_rate <= 190:
        return "very_high"
    else:
        return "Unknown"


def create_rest_blood_pressure_groups(rest_blood_pressure):
    if rest_blood_pressure <= 120:
        return "normal"
    if 120 < rest_blood_pressure <= 130:
        return "elevated"
    if 130 < rest_blood_pressure <= 140:
        return "hypertension stage_1"
    if 140 < rest_blood_pressure < 180:
        return "hypertension stage_2"
    if 180 <= rest_blood_pressure:
        return "hypertension crisis"


def change_disease_to_number(disease):
    if disease == 'positive':
        return 1
    if disease == 'negative':
        return 0


def change_blood_sugar_to_number(blood_sugar):
    if blood_sugar == True:
        return 1
    if blood_sugar == False:
        return 0
    else:
        return "Unknown"


def change_exercice_angina_to_number(exercice_angina):
    if exercice_angina == "yes":
        return 1
    if exercice_angina == "no":
        return 0
    else:
        return "Unknown"


def change_rest_electro_to_letter(rest_electro):
    if rest_electro == "normal":
        return "N"
    if rest_electro == "st_t_wave_abnormality":
        return "S"
    if rest_electro == "left_vent_hyper":
        return "L"
    else:
        return "Unknown"


def change_chest_pain_type_to_letter(chest_pain_type):
    if chest_pain_type == "asympt":
        return "A"
    if chest_pain_type == "atyp_angina":
        return "AA"
    if chest_pain_type == "non_anginal":
        return "NA"
    if chest_pain_type == "typ_angina":
        return "TA"
    else:
        return "Unknown"


def prepare_data_lable(df, train_set=True):
    df["Disease"] = df.disease.apply(change_disease_to_number)
    df.drop(["disease"], axis=1, inplace=True)

    return df


def prepare_data(df, train_set=True):
    # create new feature
    df["Age_Group"] = df.age.apply(create_age_groups)
    df["Heart_Rate_Group"] = df.max_heart_rate.apply(create_heart_rate_groups)
    df["Disease"] = df.disease.apply(change_disease_to_number)
    df["Blood_Sugar"] = df.blood_sugar.apply(change_blood_sugar_to_number)
    df["Exercice_Angina"] = df.exercice_angina.apply(change_exercice_angina_to_number)
    df["Rest_Electro"] = df.rest_electro.apply(change_rest_electro_to_letter)
    df["Chest_Pain_Type"] = df.chest_pain_type.apply(change_chest_pain_type_to_letter)
    df["Rest_Blood_Pressure"] = df.rest_blood_pressure.apply(create_rest_blood_pressure_groups)

    # drop features that we are not going to use
    df.drop([
        "age",
        "disease",
        "blood_sugar",
        "exercice_angina",
        "rest_electro",
        "chest_pain_type",
        "max_heart_rate",
        "rest_blood_pressure"], axis=1, inplace=True)

    return df


def replace_strings(df):
    df.Age_Group.replace({"young": 0, "man": 1, "old": 2}, inplace=True)
    df.Rest_Electro.replace({"N": 0, "S": 1, "L": 2, "Unknown": 3}, inplace=True)
    df.Chest_Pain_Type.replace({"A": 0, "AA": 1, "NA": 2, "TA": 3}, inplace=True)
    df.Heart_Rate_Group.replace({"very_low": 0, "low": 1, "moderate": 2, "high": 3, "very_high": 4}, inplace=True)
    df.Rest_Blood_Pressure.replace(
        {"normal": 0, "elevated": 1, "hypertension stage_1": 2, "hypertension stage_2": 3, "hypertension crisis": 4},
        inplace=True)

    return df