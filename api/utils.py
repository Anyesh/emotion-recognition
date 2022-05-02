

def response_interface(data, msg="API ran succesfully!!", status=200, explainer=None):
    if explainer:
        explainer = explainer.as_html()
    return {
        "data": {},
        "msg": msg,
        "status": status,
        "explainer": explainer,
        'data': data,
    }