"""empty message

Revision ID: v3.4.0.a
Revises: v3.2.0.a
Create Date: 2023-07-28 16:13:21.188604

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "v3.4.0.a"
down_revision = "v3.2.0.a"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("trial_intermediate_values", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "intermediate_value_index",
                sa.Integer(),
                nullable=False,
                server_default="0",
            )
        )
        batch_op.create_unique_constraint(
            "uq_trial_intermediate_values", ["trial_id", "step", "intermediate_value_index"]
        )


def downgrade():
    with op.batch_alter_table("trial_intermediate_values", schema=None) as batch_op:
        batch_op.drop_constraint(None, type_="unique")
        batch_op.drop_column("intermediate_value_index")
