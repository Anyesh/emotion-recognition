from .services.predictor import predict, home, logs
from .services.views import (
    add_model_names,
    view_models,
    update_model_name,
    remove_model,
    update_model_output,
    view_model_outputs,
    remove_model_output,
)


def app_routes(app):
    """
    """

    app.add_url_rule(rule="/", view_func=home, methods=["GET"])
    app.add_url_rule(rule="/logs", view_func=logs, methods=["GET"])

    app.add_url_rule(rule="/api/v1/predict", view_func=predict, methods=["POST"])
    app.add_url_rule(
        rule="/api/v1/add-model", view_func=add_model_names, methods=["POST"]
    )
    app.add_url_rule(
        rule="/api/v1/fetch-models", view_func=view_models, methods=["GET"]
    )
    app.add_url_rule(
        rule="/api/v1/update-model/<model_id>",
        view_func=update_model_name,
        methods=["POST"],
    )
    app.add_url_rule(
        rule="/api/v1/remove-model/<model_id>",
        view_func=remove_model,
        methods=["POST"],
    )

    app.add_url_rule(
        rule="/api/v1/view-model-outpus", view_func=view_model_outputs, methods=["GET"]
    )
    app.add_url_rule(
        rule="/api/v1/update-model-output/<model_id>",
        view_func=update_model_output,
        methods=["POST"],
    )
    app.add_url_rule(
        rule="/api/v1/remove-model-output/<model_id>",
        view_func=remove_model_output,
        methods=["DELETE"],
    )

    return app
