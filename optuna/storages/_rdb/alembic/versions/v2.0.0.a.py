"""empty message

Revision ID: v2.0.0.a
Revises: v1.3.0.a
Create Date: 2020-06-23 17:32:57.341978

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "v2.0.0.a"
down_revision = "v1.3.0.a"
branch_labels = None
depends_on = None


def upgrade():
    if op.get_context().dialect.name == "sqlite":
        # As of Alembic 1.4.2, recreation is not invoked by `add_column`.
        # Recreation must be enforced here since the added column has a non-constant default and
        # this is not supported by SQLite.
        recreate = "always"
    else:
        # For non-SQLite dialects, the default "auto" value is used.
        recreate = "auto"

    with op.batch_alter_table("trials", recreate=recreate) as batch_op:
        batch_op.add_column(
            sa.Column(
                "datetime_last_update",
                sa.DateTime(),
                server_default=sa.func.now(),
                onupdate=sa.func.now(),
                nullable=False,
            )
        )


def downgrade():
    with op.batch_alter_table("trials") as batch_op:
        batch_op.drop_column("datetime_last_update")
