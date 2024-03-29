import arxiv
import openai
import time
import json
import os
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

arxiv_client = arxiv.Client()
openai_client = openai.OpenAI()
model = "gpt-3.5-turbo"

def get_articles(topic):
    search = arxiv.Search(
    query = topic,
    max_results = 5
    )
    results = arxiv_client.results(search)
    abstracts = [(i+1, r.summary.replace("\n"," "),r.entry_id) for i,r in enumerate(results)]
    return abstracts


class ResearchSummarizer:
    thread_id = "thread_c97Apl6FpGOzbNdRRHzekDPM"
    assistant_id = "asst_mSyDDFEKMFGPoQKSKhRbYV0b"

    def __init__(self, model: str = model):
        self.openai_client = openai_client
        self.model = model
        self.assistant = None
        self.thread = None
        self.run = None
        self.summary = None

        if ResearchSummarizer.assistant_id:
            self.assistant = self.openai_client.beta.assistants.retrieve(
                assistant_id=ResearchSummarizer.assistant_id
            )
        if ResearchSummarizer.thread_id:
            self.thread = self.openai_client.beta.threads.retrieve(
                thread_id=ResearchSummarizer.thread_id
            )
        
    def create_assistant(self, name, instructions, tools):
        if not self.assistant:
            print(f"Assistant not found creating new one.")
            assistant_obj = self.openai_client.beta.assistants.create(
                name=name,
                instructions=instructions,
                tools=tools,
                model=self.model
            )
            ResearchSummarizer.assistant_id = assistant_obj.id
            self.assistant = assistant_obj
            print(f"Assist ID -- {self.assistant.id}")
    
    def create_thread(self):
        if not self.thread:
            print(f"Thread not found creating new one.")
            thread_obj = self.openai_client.beta.threads.create()
            ResearchSummarizer.thread_id = thread_obj.id
            self.thread = thread_obj
            print(f"Thread ID -- {self.thread.id}")

    def add_message_to_thread(self, role, content):
        if self.thread:
            self.openai_client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role=role,
                content=content
            )

    def run_assistant(self, instructions):
        if self.thread and self.assistant:
            self.run = self.openai_client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
                instructions=instructions
            )

    def process_message(self):
        if self.thread:
            messages = self.openai_client.beta.threads.messages.list(
                thread_id=self.thread.id
            )
            summary = []

            last_message = messages.data[0]
            role = last_message.role
            response = last_message.content[0].text.value
            summary.append(response)

            self.summary = "\n".join(summary)
            print(f"Summary -- {role.capitalize()}: ==> {response}")

    def call_required_functions(self, required_actions):
        if not self.run:
            return
        tool_outputs = []
        for action in required_actions["tool_calls"]:
            func_name = action["function"]["name"]
            arguments = json.loads(action["function"]["arguments"])

            if func_name == "get_articles":
                output = get_articles(topic=arguments["topic"])
                # print(f"ARTICLES ---- {output}")
                final_str = ""
                for item in output:
                    final_str += "["+str(item[0])+"] "+str(item[2])+"\n"+item[1]+"\n\n"
                print(f"FINAL STR ---- {final_str}")
        
                tool_outputs.append({"tool_call_id": action["id"],
                                     "output": final_str
                                     })
            else:
                raise ValueError(f"Unknown function: {func_name}")   
                             
        print("Submitting outputs to the Assistant")
        self.openai_client.beta.threads.runs.submit_tool_outputs(
            thread_id=self.thread.id,
            run_id=self.run.id,
            tool_outputs=tool_outputs
        )

    def get_summary(self):
        return self.summary
    
    def wait_for_completion(self):
        if self.thread and self.run:
            while True:
                time.sleep(5)
                run_status = self.openai_client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=self.run.id
                )
                print(f"RUN STATUS -- {run_status.model_dump_json(indent=4)}")

                if run_status.status == "completed":
                    self.process_message()
                    break
                elif run_status.status == "requires_action":
                    print("FUNCTION CALLING NOW")
                    self.call_required_functions(
                        required_actions=run_status.required_action.submit_tool_outputs.model_dump()
                    )

    def run_steps(self):
        run_steps = self.openai_client.beta.threads.runs.steps.list(
            thread_id=self.thread.id,
            run_id=self.run.id
        )
        print(f"RUN STEPS--- {run_steps}")

def main():
    # test_topic1 = "attention mechanisms"
    # test_topic2 = "LLM Context Length Extension",
    # abstracts = get_articles(topic=test_topic1)
    # print(abstracts)

    summarizer = ResearchSummarizer()

    st.title("Research Topic Summarizer")
    with st.form(key="user_input_form"):
        instructions = st.text_input("Enter Topic:")
        submit_button = st.form_submit_button(label="Run Assitant")

        if submit_button:
            summarizer.create_assistant(
                name="Research Topic Summarizer",
                instructions="You are a helpful assistant that creates an abstract on the topic by referring to the research articles given by the user. You will be given a numbered list of research articles and reference links. Please add references using the given links.",
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_articles",
                            "description": "Get the research papers from arxiv",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "topic": {
                                        "type": "string",
                                        "description": "Research topic"
                                    }
                                },
                                "required": ["topic"]
                            }
                        }
                    }
                ],
            )

            summarizer.create_thread()

            summarizer.add_message_to_thread(
                role="user",
                content=f"Create an abstract on the topic by referring to these research articles. \n{instructions}. Please add references using the given links."
            )

            summarizer.run_assistant(instructions="Create an article on the topic by referring to these research articles. Please add references using the given links.")

            summarizer.wait_for_completion()
            summary = summarizer.get_summary()

            st.write(summary)
            # st.text(summary)
            
            # st.text("Run Steps:")
            # st.code(summarizer.run_steps())

if __name__ == "__main__":
    main()