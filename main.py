from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScoringItem(BaseModel):
    
    itching: int
    skin_rash: int
    nodal_skin_eruptions: int
    patches_in_throat: int
    sweating: int
    dehydration: int
    yellowish_skin: int
    bruising: int
    drying_and_tingling_lips: int
    toxic_look_typhos: int
    red_spots_over_body: int
    dischromic_patches: int
    pus_filled_pimples: int
    blackheads: int
    skin_peeling: int
    silver_like_dusting: int
    blister: int
    red_sore_around_nose: int
    yellow_crust_ooze: int
    inflammatory_nails: int
    small_dents_in_nails: int
    scurring: int
    painful_walking: int
    prominent_veins_on_calf: int
    muscle_pain: int
    movement_stiffness: int
    swelling_joints: int
    stiff_neck: int
    muscle_weakness: int
    hip_joint_pain: int
    knee_pain: int
    swollen_extremeties: int
    brittle_nails: int
    neck_pain: int
    weakness_in_limbs: int
    back_pain: int
    cold_hands_and_feets: int
    muscle_wasting: int
    joint_pain: int
    swollen_legs: int
    chills: int
    shivering: int
    headache: int
    dizziness: int
    cramps: int
    slurred_speech: int
    loss_of_balance: int
    unsteadiness: int
    weakness_of_one_body_side: int
    altered_sensorium: int
    coma: int
    acidity: int
    ulcers_on_tongue: int
    vomiting: int
    indigestion: int
    nausea: int
    constipation: int
    abdominal_pain: int
    diarrhoea: int
    swelling_of_stomach: int
    pain_during_bowel_movements: int
    pain_in_anal_region: int
    bloody_stool: int
    irritation_in_anus: int
    passage_of_gases: int
    belly_pain: int
    abnormal_menstruation: int
    stomach_bleeding: int
    distention_of_abdomen: int
    stomach_pain: int
    continuous_sneezing: int
    cough: int
    breathlessness: int
    phlegm: int
    throat_irritation: int
    redness_of_eyes: int
    sinus_pressure: int
    runny_nose: int
    congestion: int
    mucoid_sputum: int
    rusty_sputum: int
    blood_in_sputum: int
    lack_of_concentration: int
    irritability: int
    depression: int
    lethargy: int
    restlessness: int
    mood_swings: int
    anxiety: int
    burning_micturition: int
    spotting_urination: int
    dark_urine: int
    yellow_urine: int
    bladder_discomfort: int
    foul_smell_of_urine: int
    continuous_feel_of_urine: int
    polyuria: int
    palpitations: int
    enlarged_thyroid: int
    swollen_blood_vessels: int
    fast_heart_rate: int
    chest_pain: int
    swelled_lymph_nodes: int
    fatigue: int
    weight_gain: int
    weight_loss: int
    irregular_sugar_level: int
    high_fever: int
    sunken_eyes: int
    loss_of_appetite: int
    mild_fever: int
    yellowing_of_eyes: int
    acute_liver_failure: int
    fluid_overload: int
    malaise: int
    obesity: int
    puffy_face_and_eyes: int
    extra_marital_contacts: int
    spinning_movements: int
    internal_itching: int
    watering_from_eyes: int
    family_history: int
    fluid_overload_1: int
    history_of_alcohol_consumption: int
    receiving_unsterile_injections: int
    receiving_blood_transfusion: int
    pain_behind_the_eyes: int
    blurred_and_distorted_vision: int
    excessive_hunger: int
    loss_of_smell: int
    increased_appetite: int
    visual_disturbances: int


# loading the saved model
model = pickle.load(open('Random_Forest_Classifier.pkl', 'rb'))


import pandas as pd
@app.post('/')
async def scoring_endpoint(item:ScoringItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())# np.array([[item.dict().values()]])
    yhat = model.predict(df)

    return int(yhat)