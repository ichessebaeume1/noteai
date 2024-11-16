import ollama

class Summarizer:
    def __init__(self, topic):
        self.topic = topic
        self.memory = ""

    def summarize_chunks(self, batch):
        prompt = f"Create 1-2 bullet points on the following text focusing on {self.topic}: {batch}"

        # Generate Summary
        print("Starting summary...")
        response = ollama.chat(model="llama3.1", messages=[{'role': 'user', 'content': prompt}])['message']['content']

        self.update_memory(response)
        return response

    def update_memory(self, summary):
        """Adds the latest summary to memory."""
        self.memory += " " + summary

    def final_review(self):
        inputs = self.tokenizer([f"Summarize the following transcript in focus on {self.topic}: {self.memory}"], max_length=1024, return_tensors="pt")

        # Generate Summary
        print("Starting final summary...")
        summary_ids = self.model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=150)
        final_summary = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print("Final Summary:", final_summary)