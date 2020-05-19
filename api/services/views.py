import json
import ast
from bson.json_util import dumps
from flask import request, Response
from ..settings import shared_components
from bson.objectid import ObjectId


def add_model_names():
    """
    """

    db = shared_components["db"]
    collection = db.models
    try:
        # Create new users
        if request.json:
            body = request.json
        else:
            raise TypeError("Invalid Request Format")

        record_id = collection.insert(body)
        return Response("Successfully created the resource", status=201)

    except Exception as e:
        # Error while trying to create the resource
        print("Exception: {}".format(e))
        return Response("Error while trying to create resource")


def update_model_name(model_id):
    """

    """

    db = shared_components["db"]
    collection = db.models
    try:
        # Get the value which needs to be updated
        if request.json:
            body = ast.literal_eval(json.dumps(request.json))
        else:
            raise TypeError("Invalid request format")

        # Updating the user
        records_updated = collection.update_one(
            {"_id": ObjectId(model_id)}, {"$set": body}
        )

        # Check if resource is updated
        if records_updated.modified_count > 0:
            # Prepare the response as resource is updated successfully
            return Response("Resource updated successfully", status=200)
        else:
            # Bad request as the resource is not available to update
            # Add message for debugging purpose
            return Response("Resource not available", status=404)
    except Exception as e:
        # Error while trying to update the resource
        # Add message for debugging purpose
        print("Exception: {}".format(e))
        return Response("Error while updating the resource", status=500)


def view_models():
    """

    """
    db = shared_components["db"]
    collection = db.models
    try:
        # Fetch all the record(s)
        records_fetched = collection.find({})

        # Check if the records are found
        if records_fetched.count() > 0:
            # Prepare the response
            records = dumps(records_fetched)
            resp = Response(records, status=200, mimetype="application/json")
            return resp
        else:
            # No records are found
            return Response("No records are found", status=404)
    except Exception as e:
        print("Exception: {}".format(e))
        # Error while trying to fetch the resource
        return Response("Error while trying to fetch the resource", status=500)


def remove_model(model_id):
    """
    """
    db = shared_components["db"]
    collection = db.models
    try:
        # Delete the user
        delete_model = collection.delete_one({"_id": ObjectId(model_id)})

        if delete_model.deleted_count > 0:
            # Prepare the response
            return Response("Resource deleted successfully", status=200)
        else:
            # Resource Not found
            return Response("Resource Not Found", status=404)
    except Exception as e:
        # Error while trying to delete the resource
        # Add message for debugging purpose
        print("Exception: {}".format(e))
        return Response("Resource deletion failed", status=500)


def update_model_output(model_id):
    """

    """

    db = shared_components["db"]
    collection = db.outputs
    try:
        # Get the value which needs to be updated
        if request.json:
            body = ast.literal_eval(json.dumps(request.json))
        else:
            raise TypeError("Invalid request format")

        # Updating the user
        records_updated = collection.update_one(
            {"_id": ObjectId(model_id)}, {"$set": body}
        )

        # Check if resource is updated
        if records_updated.modified_count > 0:
            # Prepare the response as resource is updated successfully
            return Response("Resource updated successfully", status=200)
        else:
            # Bad request as the resource is not available to update
            # Add message for debugging purpose
            return Response("Resource not available", status=404)
    except Exception as e:
        # Error while trying to update the resource
        # Add message for debugging purpose
        print("Exception: {}".format(e))
        return Response("Error while updating the resource", status=500)


def view_model_outputs():
    """

    """
    db = shared_components["db"]
    collection = db.outputs
    try:
        # Fetch all the record(s)
        records_fetched = collection.find({})

        # Check if the records are found
        if records_fetched.count() > 0:
            # Prepare the response
            records = dumps(records_fetched)
            resp = Response(records, status=200, mimetype="application/json")
            return resp
        else:
            # No records are found
            return Response("No records are found", status=404)
    except Exception as e:
        print("Exception: {}".format(e))
        # Error while trying to fetch the resource
        return Response("Error while trying to fetch the resource", status=500)


def remove_model_output(model_id):
    """
    """
    db = shared_components["db"]
    collection = db.outputs
    try:
        # Delete the user
        delete_model = collection.delete_one({"_id": ObjectId(model_id)})

        if delete_model.deleted_count > 0:
            # Prepare the response
            return Response("Resource deleted successfully", status=200)
        else:
            # Resource Not found
            return Response("Resource Not Found", status=404)
    except Exception as e:
        # Error while trying to delete the resource
        # Add message for debugging purpose
        print("Exception: {}".format(e))
        return Response("Resource deletion failed", status=500)
