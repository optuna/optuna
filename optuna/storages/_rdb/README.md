### Create migration script

If you want to introduce a schematic change to Optuna storage,
you need to create an alembic revision file.

### 1. Checkout the current master branch

### 2. Create a study with the name specified by `alembic.ini`

You can check the name that alembic expects by executing:

```sh
$ grep 'sqlalchemy.url' alembic.ini
sqlalchemy.url = sqlite:///alembic.db
```

Let's create storage on `alembic.db`.

```sh
$ optuna create-study --storage sqlite:///alembic.db
[I 2022-03-03 16:44:06,783] A new study created in RDB with name: no-name-f72bf92f-9864-4159-941c-c8c7de027fed
no-name-f72bf92f-9864-4159-941c-c8c7de027fed
```

### 3. Run the Alembic command to generate a revision file. Please change the revision id based on the Optuna version.

```sh
$ alembic revision --autogenerate --rev-id v3.0.0.a
INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
  Generating /Users/himkt/work/github.com/himkt/optuna/optuna/storages/_rdb/alembic/versions/v3.0.0.a_.py ...  done
```

If `v3.0.0.a` already exists, you can create another revision by changing the suffix: `v3.0.0.b`.

### 4. Write migration logic

You can read existing revisions on GitHub.
(e.g. https://github.com/optuna/optuna/pull/2395)

### 5. Run migration to check if it works as expected

```sh
$ optuna storage upgrade --storage sqlite:///alembic.db
[I 2022-03-03 16:33:52,271] Upgrading the storage schema to the latest version.
[I 2022-03-03 16:33:52,285] Completed to upgrade the storage.
```

Please write tests as well. It is important for users to be able to upgrade storage smoothly.
(e.g. https://github.com/optuna/optuna/pull/3113)

