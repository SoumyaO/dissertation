import openai

openai.api_key = 'xx'

training_file_id='file-G7g0OJhc7mas6CV8iswCMu8j'
validation_file_id='file-aA2mGNNAyHXKBTCKlPKbQKwW'

suffix_name = "arp-finetune"

# Create a fine-tuning job
response = openai.FineTuningJob.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model="gpt-3.5-turbo",
    suffix=suffix_name,
)

job_id = response["id"]

print(response)

response = openai.FineTuningJob.retrieve(job_id)
print(response)

response = openai.FineTuningJob.list_events(id=job_id, limit=50)

events = response["data"]
events.reverse()

for event in events:
    print(event["message"])

response = openai.FineTuningJob.retrieve(job_id)
fine_tuned_model_id = response["fine_tuned_model"]

print(response)
print("\nFine-tuned model id:", fine_tuned_model_id)
