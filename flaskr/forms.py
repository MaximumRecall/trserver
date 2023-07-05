from flask_wtf import FlaskForm
from wtforms import StringField, HiddenField
from wtforms.validators import DataRequired

class SearchForm(FlaskForm):
    user_id_str = HiddenField(validators=[DataRequired()])
    search_text = StringField(validators=[DataRequired()])
