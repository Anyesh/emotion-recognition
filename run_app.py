from api.app import app
from api import settings

if __name__ == "__main__":

    app.run(debug=settings.DEBUG)
