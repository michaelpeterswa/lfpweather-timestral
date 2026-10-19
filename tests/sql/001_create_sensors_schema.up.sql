-- Vendored verbatim from timescale-migrations-timestral/migrations/001_create_sensors_schema.up.sql
-- Ownership is left to whichever role runs the migration rather than named
-- explicitly. This statement previously read AUTHORIZATION timescale, which
-- cannot succeed on a database where that role does not exist -- and the
-- production database has only postgres and grafana. That is what left it
-- stuck at version 1 dirty, with every table since applied by hand.
CREATE SCHEMA IF NOT EXISTS sensors;
