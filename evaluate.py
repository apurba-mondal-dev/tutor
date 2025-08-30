# --- Step 1: Import all necessary libraries ---
import pandas as pd
import gradio as gr
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from transformers import pipeline

print("--- FINAL AI Tutor Analysis Dashboard Starting Up ---")
print("This may take a few minutes to download models and process all datasets...")

# --- Step 2: Load and process all the JSON datasets ---
def load_and_process_data(filename):
    """Loads a JSON file and flattens it for training/testing."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{filename}' not found. Please upload it.")
        return None

    records = []
    for conversation in data:
        history = conversation.get('conversation_history', '')
        expert_response_data = conversation.get('tutor_responses', {}).get('Expert', None)
        
        if history and expert_response_data:
            expert_response_text = expert_response_data.get('response', '')
            # Combine history and response for better training context
            combined_text = f"CONTEXT:\n{history}\n\nRESPONSE TO CLASSIFY:\n{expert_response_text}"
            
            records.append({
                'conversation_id': conversation.get('conversation_id'),
                'combined_text': combined_text,
                'student_question': history.split('Student:')[0].replace('Tutor:  Hi, could you please provide a step-by-step solution for the question below? The question is:', '').strip(),
                'expert_response': expert_response_text,
                'identifies_mistake': expert_response_data.get('annotation', {}).get('Mistake_Identification', 'No'),
                'provides_guidance': expert_response_data.get('annotation', {}).get('Providing_Guidance', 'No')
            })
            
    return pd.DataFrame(records)

print("Loading all three datasets...")
train_df_part1 = load_and_process_data('trainset.json')
train_df_part2 = load_and_process_data('dev_testset.json')
final_test_df = load_and_process_data('testset.json')

if train_df_part1 is None or train_df_part2 is None or final_test_df is None:
    exit()

# --- NEW: Combine trainset.json and dev_testset.json into one large training set ---
combined_train_df = pd.concat([train_df_part1, train_df_part2], ignore_index=True)
print(f"Created a combined training set with {len(combined_train_df)} examples.")
print(f"Using the original testset.json with {len(final_test_df)} examples for the evaluation tab.")


# --- Step 3: Encode labels into numbers ---
le_mistake = LabelEncoder()
le_guidance = LabelEncoder()
# Fit on the combined data to ensure all possible labels are learned
le_mistake.fit(combined_train_df['identifies_mistake'])
le_guidance.fit(combined_train_df['provides_guidance'])

combined_train_df['identifies_mistake_encoded'] = le_mistake.transform(combined_train_df['identifies_mistake'])
combined_train_df['provides_guidance_encoded'] = le_guidance.transform(combined_train_df['provides_guidance'])
print("Text labels converted to numbers.")

# --- Step 4: Load AI Models ---
print("Loading Sentence-BERT model (all-mpnet-base-v2)...")
embedding_model = SentenceTransformer('all-mpnet-base-v2') 

print("Loading Zero-Shot Classification model (facebook/bart-large-mnli)...")
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


# --- Step 5: Create Embeddings and Train XGBoost Models on ALL available data ---
print("Creating embeddings for the large training dataset (this will take a while)...")
train_embeddings = embedding_model.encode(combined_train_df['combined_text'].tolist(), show_progress_bar=True)

print("\n--- Training model for 'Mistake Identification' ---")
xgb_classifier_mistake = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_classifier_mistake.fit(train_embeddings, combined_train_df['identifies_mistake_encoded'])
print("Mistake Identification model trained.")

print("\n--- Training model for 'Providing Guidance' ---")
xgb_classifier_guidance = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_classifier_guidance.fit(train_embeddings, combined_train_df['provides_guidance_encoded'])
print("Providing Guidance model trained.")


# --- Step 6: Define the backend functions for the UI ---

# Function for the "Live Playground" tab (XGBoost)
def classify_with_xgboost(student_question, tutor_answer):
    if not student_question or not tutor_answer:
        return {}, {}
    combined_input = f"CONTEXT:\n{student_question}\n\nRESPONSE TO CLASSIFY:\n{tutor_answer}"
    input_embedding = embedding_model.encode([combined_input])
    
    mistake_probs = xgb_classifier_mistake.predict_proba(input_embedding)[0]
    mistake_confidences = {label: float(prob) for label, prob in zip(le_mistake.classes_, mistake_probs)}
    
    guidance_probs = xgb_classifier_guidance.predict_proba(input_embedding)[0]
    guidance_confidences = {label: float(prob) for label, prob in zip(le_guidance.classes_, guidance_probs)}
    
    return mistake_confidences, guidance_confidences

# Function for the "Test Set Evaluation" tab
def evaluate_from_testset(conversation_id):
    if not conversation_id:
        return "", "", {}, {}, "", ""
    # Look up the conversation in our final_test_df
    record = final_test_df[final_test_df['conversation_id'] == conversation_id].iloc[0]
    combined_text_for_eval = record['combined_text']
    expert_answer = record['expert_response']
    
    # Get ground truth labels
    ground_truth_mistake = record['identifies_mistake']
    ground_truth_guidance = record['provides_guidance']
    
    input_embedding = embedding_model.encode([combined_text_for_eval])
    mistake_probs = xgb_classifier_mistake.predict_proba(input_embedding)[0]
    mistake_confidences = {label: float(prob) for label, prob in zip(le_mistake.classes_, mistake_probs)}
    
    guidance_probs = xgb_classifier_guidance.predict_proba(input_embedding)[0]
    guidance_confidences = {label: float(prob) for label, prob in zip(le_guidance.classes_, guidance_probs)}

    return combined_text_for_eval, expert_answer, mistake_confidences, guidance_confidences, ground_truth_mistake, ground_truth_guidance


# --- Step 7: Launch the final, tabbed web interface ---
print("\n--- All models loaded. Launching Interactive Web Interface ---")

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎓 AI Tutor Analysis Dashboard (Final Version)")
    
    with gr.Tabs():
        # --- First Tab: Live Playground ---
        with gr.TabItem("Live Playground"):
            gr.Markdown("Enter any student question and tutor answer to get a live classification from the AI.")
            with gr.Row():
                with gr.Column(scale=2):
                    student_input_live = gr.Textbox(lines=5, label="Student's Question / Problem")
                    tutor_input_live = gr.Textbox(lines=5, label="Tutor's Answer to Classify")
                    submit_btn_live = gr.Button("Classify Tutor's Answer", variant="primary")
                    gr.Examples(
                        examples=combined_train_df[['student_question', 'expert_response']].head(3).values.tolist(),
                        inputs=[student_input_live, tutor_input_live]
                    )
                with gr.Column(scale=1):
                    gr.Markdown("### Classification Results")
                    mistake_output_live = gr.Label(label="Identifies Student's Mistake?", num_top_classes=3)
                    guidance_output_live = gr.Label(label="Provides Helpful Guidance?", num_top_classes=3)
            
            submit_btn_live.click(
                fn=classify_with_xgboost, 
                inputs=[student_input_live, tutor_input_live], 
                outputs=[mistake_output_live, guidance_output_live]
            )

        # --- Second Tab: Test Set Evaluation ---
        with gr.TabItem("Test Set Evaluation"):
            gr.Markdown("Select a conversation from the hold-out `testset.json` file to compare the AI's prediction against the correct answer.")
            with gr.Row():
                with gr.Column(scale=2):
                    conversation_selector = gr.Dropdown(
                        choices=final_test_df['conversation_id'].tolist(),
                        label="Select a Conversation ID from the Test Set"
                    )
                    submit_btn_eval = gr.Button("Evaluate AI Performance", variant="primary")
                    gr.Markdown("### Conversation Context")
                    history_output_eval = gr.Textbox(lines=8, label="Full Context Given to AI", interactive=False)
                    expert_output_eval = gr.Textbox(lines=3, label="Expert's Response in this Conversation", interactive=False)
                with gr.Column(scale=1):
                    gr.Markdown("### AI Prediction (with Confidence %)")
                    mistake_pred_output_eval = gr.Label(label="Predicted: Identifies Mistake?", num_top_classes=3)
                    guidance_pred_output_eval = gr.Label(label="Predicted: Provides Guidance?", num_top_classes=3)
                    gr.Markdown("---")
                    gr.Markdown("### Ground Truth (Correct Answer)")
                    mistake_truth_output_eval = gr.Textbox(label="Actual: Identifies Mistake?", interactive=False)
                    guidance_truth_output_eval = gr.Textbox(label="Actual: Provides Guidance?", interactive=False)

            submit_btn_eval.click(
                fn=evaluate_from_testset, 
                inputs=conversation_selector, 
                outputs=[history_output_eval, expert_output_eval, mistake_pred_output_eval, guidance_pred_output_eval, mistake_truth_output_eval, guidance_truth_output_eval]
            )

demo.launch()