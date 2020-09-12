{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   ..
      Ordinary methods

{% for item in methods %}
{% if item not in ('__init__',) %}
   .. automethod:: {{ item }}
{% endif %}
{%- endfor %}

   ..
      Attributes

{% block attributes %} {% if attributes %}

   .. rubric:: Attributes

{% for item in attributes %}
   .. autoattribute:: {{ item }}
{%- endfor %}
{% endif %} {% endblock %}
