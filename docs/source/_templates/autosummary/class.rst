{% extends "!autosummary/class.rst" %}

{#
An autosummary template to exclude the class constructor (__init__)
which doesn't contain any docstring in Optuna.
#}

{% block methods %}
   {% set methods = methods | select("ne", "__init__") | list %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}

{% endblock %}
