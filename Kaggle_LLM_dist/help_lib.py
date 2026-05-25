import re
import warnings


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
EXPRESSION_RESULT_RE = re.compile(r"(?<!\S)[\d\s()+\-*/]+\s*=\s*-?\d+(?!\S)")
THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)
MATH_EXPRESSION_RE = re.compile(r'(?:[\(\d][\d\+\-\*\/\(\)\. \t]*=[\d\+\-\*\/\(\)\. \t=]*[\d\)])')
PREFIX_RE = re.compile(r'^(\d+[\.\)]\s+|Step\s*\d+\s*[:\-\.]+\s*|[\[\(]\d+[\]\)]\s*)')



def messages_to_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def safe_eval_expr(expr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        
        allowed_globals = {"__builtins__": {}}
        allowed_locals = {}
        result = eval(expr, allowed_globals, allowed_locals)
        
    return result

def extract_numbers_from_text(text: str):  # Достаёт только числа. Не учитывает - перед числами. Это и не надо, потому что минус всегда можно внести
    nums = []
    current = []

    for ch in text:
        if ch.isdigit():
            current.append(ch)
        else:
            if current:
                nums.append(int("".join(current)))
                current = []
    if current:
        nums.append(int("".join(current)))

    return nums


def is_valid_equation(candidate: str, nums: list[int], target: int, tol=1e-9) -> bool:
    if "=" not in candidate:
        return False

    left, right = candidate.split("=", 1)
    left = left.strip()
    right = right.strip()

    try:
        got = safe_eval_expr(left)
        rhs = float(right)
        tgt = float(target)
    except Exception:
        return False

    if abs(got - rhs) > tol:
        return False

    if abs(rhs - tgt) > tol:
        return False

    
    used_nums = extract_numbers_from_text(left)
    return sorted(used_nums) == sorted(int(x) for x in nums)



def extract_last_answer(text):
    answers = ANSWER_RE.findall(text)
    return answers[-1] if answers else "" # Берём самый последний ответ, потому что думаем, что он финальный

def extract_generated_equation(text):
    text = text.strip()
    matches = list(EXPRESSION_RESULT_RE.finditer(text))
    if not matches:
        return ""
    return matches[-1].group(0).strip() # Берём последнее вхождение, поотому что скорее всего до этого могли быть минимальные рассуждения

def normalize_text(s: str) -> str:
    return " ".join(s.strip().split())

def collate_data_fn(samples):
    return {
        "messages": [s["messages"] for s in samples],
        "nums": [s["nums"] for s in samples],
        "target": [s["target"] for s in samples]
    }

