# Documentation Overview

This folder contains the complete documentation for the LBP package AIXD architecture.

---

## Core Documents

### 1. [QUICK_START.md](QUICK_START.md)
**For**: New users getting started  
**Contains**:
- 5-minute workflow guide
- Complete code examples
- Common operations
- Phase 7 API patterns

**Start here** if you're new to the package.

---

### 2. [SEPARATION_OF_CONCERNS.md](SEPARATION_OF_CONCERNS.md)
**For**: Understanding architecture and component responsibilities  
**Contains**:
- Architecture layers diagram
- Component responsibilities (what each class does/doesn't do)
- Ownership patterns
- Key design patterns
- Anti-patterns to avoid
- Validation points

**Read this** to understand the overall architecture.

---

### 3. [CORE_DATA_STRUCTURES.md](CORE_DATA_STRUCTURES.md)
**For**: Understanding the data model  
**Contains**:
- DataObjects (typed primitives)
- DataBlocks (value collections)
- DatasetSchema (structure definition)
- SchemaRegistry (hash-to-ID mapping)
- ExperimentData (single experiment)
- Dataset (experiment collection)
- Serialization formats
- Validation hierarchy

**Read this** to understand how data is structured and validated.

---

### 4. [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
**For**: Understanding what has been implemented and how it evolved  
**Contains**:
- Evolution timeline (Phases 1-7)
- Current architecture
- Code statistics
- Breaking changes from previous versions
- Benefits of current architecture
- Known issues & limitations
- Next steps

**Read this** to understand the implementation history and current status.

---

## Document Relationships

```
New User Flow:
  QUICK_START.md
    └─> Start coding with examples
    └─> Need architecture details?
         └─> SEPARATION_OF_CONCERNS.md
              └─> Need data model details?
                   └─> CORE_DATA_STRUCTURES.md

Developer Flow:
  SEPARATION_OF_CONCERNS.md (understand responsibilities)
    └─> CORE_DATA_STRUCTURES.md (understand data model)
         └─> IMPLEMENTATION_SUMMARY.md (understand evolution)
              └─> QUICK_START.md (see it in action)
```

---

## Quick Reference

**I want to...**

- **Start using the package** → [QUICK_START.md](QUICK_START.md)
- **Understand the architecture** → [SEPARATION_OF_CONCERNS.md](SEPARATION_OF_CONCERNS.md)
- **Understand the data model** → [CORE_DATA_STRUCTURES.md](CORE_DATA_STRUCTURES.md)
- **See implementation history** → [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

---

## Additional Resources

- **Example code**: `../examples/aixd_example.py`
- **Core module docs**: `../src/lbp_package/core/README.md`
- **Main README**: `../README.md`
- **Tests**: `../tests/`

---

All documents are up-to-date as of **November 25, 2025** and reflect the **Phase 7 API**.
