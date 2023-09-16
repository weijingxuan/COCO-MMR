from dataclasses import dataclass
from typing import List, Optional

def get_question_text(problem):
    question = problem['question']
    return question


def get_context_text(problem, use_caption):
    txt_context = problem['hint']
    img_context = problem['caption'] if use_caption else ""
    context = " ".join([txt_context, img_context]).strip()
    if context == "":
        context = "N/A"
    return context


def get_choice_text(probelm, options):
    choices = probelm['choices']
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    return choice_txt

def get_origin_answer(problem, options):
    return problem['choices'][problem['answer']]

def get_answer(problem, options):
    return options[problem['answer']]


def get_lecture_text(problem):
    lecture = problem['lecture'].replace("\n", "\\n")
    return lecture


def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    solution = problem['solution'].replace("\n", "\\n")
    return solution


def create_one_example_coco(format, question, context, choice, answer, lecture, solution, test_example=True, WithOutput = False, curr_le_data=None):

    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "Q":
        input = f"Question: {question}\n"
    elif input_format == "QG":
        if curr_le_data is not None:
            input = f"Question: {question}\n Solution: {curr_le_data}\n"
        else:
            input = f"Question: {question}\n Solution: {solution}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    elif input_format == "QM":
        input = f"Question: {question}\nOptions: {choice}\n"
    elif input_format == "QC":
        input = f"Question: {question}\nContext: {context}\n"
    elif input_format == "QCMG":
        if curr_le_data is not None:
            input = f"Question: {question}\n{curr_le_data}\n"
        else:
            input = f"Question: {question}\n{solution}\n"
    elif input_format == "CQMG":
        if curr_le_data is not None:
            input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n{curr_le_data}\n"
        else:
            input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\nSolution: {lecture} {solution}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"
    elif input_format == "QCMA":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nAnswer: The answer is {answer}.\n"
    elif input_format == "QCA":
        input = f"Question: {question}\nContext: {context}\nAnswer: The answer is {answer}. \nBECAUSE:"

    # Outputs
    if test_example:
        if output_format == 'A':
            output = "Answer:"
        elif output_format == 'E':
            output = "Solution:"
        else:
            output = "Solution:"
    elif output_format == 'A':
        # output = f"Answer: The answer is {answer}."
        output = f"Answer: {answer}"
        
    elif output_format == 'AL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == 'AE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == 'ALE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == 'AEL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == 'LA':
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == 'EA':
        output = f"Answer: {solution} The answer is {answer}."
    elif output_format == 'LEA':
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == 'ELA':
        output = f"Answer: {solution} {lecture} The answer is {answer}."

    elif output_format == 'LE':
        output = f"Solution: {lecture} {solution}."

    elif output_format == 'E':
        output = f"Solution: {solution}"
        
    
    if WithOutput:
        if output.endswith("BECAUSE:"):
            output = output.replace("BECAUSE:", "").strip()
        if output_format == 'E':
            text = input + f'Solution:'
        elif output_format == 'A':
            text = input + f'Answer:'
        else: 
            text = input + f'Solution:'
        text = text.replace("  ", " ").strip()
        output = output.replace("  ", " ").strip()
        return text, output
        
        
    text = input + output
    text = text.replace("  ", " ").strip()
    if text.endswith("BECAUSE:"):
        text = text.replace("BECAUSE:", "").strip()
    return text

def build_train_pair_coco(coco_data, image_id, args, curr_le_data=None):
    def get_image_description(coco_data, image_id):
        if "final_answer" in coco_data[image_id]:
            return coco_data[image_id]["final_answer"]
        else:
            return ""
    def get_image_question(coco_data, image_id):
        if "question" in coco_data[image_id]:
            return coco_data[image_id]["question"]
        else:
            return ""
    def get_image_Context(coco_data, image_id):
        if "Context" in coco_data[image_id]:
            return coco_data[image_id]["Context"]
        else:
            return ""
    def get_image_Context(coco_data, image_id):
        return coco_data[image_id]["Context"]
    
    examples = []
    question = get_image_question(coco_data, image_id)  #问题
    solution = get_image_Context(coco_data, image_id)  #推理
    answer = get_image_description(coco_data, image_id) #最终答案
    context=""
    choice=""
    lecture=""
    test_example, target = create_one_example_coco(args.prompt_format,
                                    question,
                                    context,
                                    choice,
                                    answer,
                                    lecture,
                                    solution,
                                    test_example=False,WithOutput = True, curr_le_data=curr_le_data)
    examples.append(test_example)
    
    # target = target.replace("Answer:", "").strip()
    # create the prompt input
    prompt_input = '\n\n'.join(examples)

    return prompt_input, target



@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    le_input_ids: List[List[int]]
    le_attention_mask: Optional[List[List[int]]]
    le_token_type_ids: Optional[List[List[int]]]
    label: Optional[int]