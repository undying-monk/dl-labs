# Repository Instructions

## Structure

* `src/` — application services
* `src/libs/` — shared libraries

## General Rules

* Follow existing code patterns.
* Keep changes small and focused.
* Do not modify unrelated code.
* Reuse existing libraries and utilities.
* Do not add dependencies unless necessary.
* Do not change public APIs without being asked.

## Scope

When working on a service:

* Focus on that service's directory.
* Read shared libraries when necessary.
* Do not modify other services unless explicitly required.

## Testing

* Add or update tests for code changes.
* Run relevant tests before finishing.
* Do not ignore test failures without explaining them.

## Safety

* Do not commit secrets.
* Do not modify production configuration unless requested.
* Do not create commits unless requested.
