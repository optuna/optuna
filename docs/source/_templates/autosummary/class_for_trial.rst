{% extends "!autosummary/class.rst" %}

{#
An autosummary template to exclude the class constructor (__init__)
which doesn't contain any docstring in Optuna.
#}

{# Maintain the list of pages to hide inherited members #}
{% set list_for_hide_inherited = [
   "optuna.trial.TrialState"
] %}

{% block methods %}
   {% set methods = methods | select("ne", "__init__") | list %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in methods %}
   {%- if fullname in list_for_hide_inherited %}
   {%- if item not in inherited_members %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {%- else %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}
{% endblock %}

{% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
   {%- if fullname in list_for_hide_inherited %}
   {%- if item not in inherited_members %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {%- else %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}
{% endblock %}