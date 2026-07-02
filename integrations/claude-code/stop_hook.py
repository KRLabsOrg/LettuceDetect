import json
from integrations.claude_code.hallucination_detector import main

def run_hook(session: Dict, settings: Dict):
    """
    Run the HallucinationDetector hook.

    Args:
    - session (Dict): The current session.
    - settings (Dict): The hook settings.
    """
    answer = session['answer']
    context = session['context']
    encoder_model = settings['encoder_model']
    main(answer, context, encoder_model)