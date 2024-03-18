Using the Fine-tuned model
---------------------------
For using the model, run -
streamlit run interface_ft.py


For Fine-tuning
---------------
The dataset files for fine-tuning are ft_train_data.jsonl and ft_valid_data.jsonl


For fine-tuning, run -
python ft_checking_dataset.py
python ft_upload_to_openai.py
python ft_finetune_openai.py
