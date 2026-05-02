
def assign_category(text):
    text = text.lower()

    if "refund" in text:
        return "refund"
    elif "payment" in text:
        return "payment"
    elif "plan" in text or "pricing" in text:
        return "pricing"
    elif "sla" in text or "support" in text:
        return "support"
    else:
        return "general"
    

def detect_category(text):
    text = text.lower()
    if "refund" in text:
        return "refund"
    elif "plan" in text or "pricing" in text:
        return "pricing"
    elif "failed payment" in text or "payment issue" in text:
        return "payment"
    elif "sla" in text or "support" in text:
        return "support"
    else:
        return "Non-Category"