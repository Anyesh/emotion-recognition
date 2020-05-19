

def response_interface(data, msg="API ran succesfully!!", status=200, explainer=None):
    if explainer:
        explainer = explainer.as_html()
    response = {"data": {}, "msg": msg,  "status": status,  "explainer": explainer}
    response['data'] = data

    return response