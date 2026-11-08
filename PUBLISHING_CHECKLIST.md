# Publishing Checklist

## Private GitHub review

- [ ] Create a private repository named `llm_reference_aes`.
- [ ] Push the contents of this directory to the `main` branch.
- [ ] Add the repository description and topics from `REPOSITORY_METADATA.md`.
- [ ] Confirm that the Python syntax-check workflow passes.
- [ ] Ask all coauthors to review the README, licenses, author names, prompts,
      generated references, and release scope.
- [ ] Confirm that no raw student responses or datasets are present.
- [ ] Decide whether to rerun the cleaned code or retain the disclosure in
      `CODE_AUDIT.md` and `RELEASE_NOTES_v1.0.0.md`.

## Public release

- [ ] Make the repository public after author approval.
- [x] Add the final repository URL to `CITATION.cff` and `README.md`.
- [ ] Create and push tag `v1.0.0`.
- [ ] Create a GitHub release using `RELEASE_NOTES_v1.0.0.md`.
- [ ] Attach `llm_reference_aes_v1.0.0.zip` to the release.

## DOI and manuscript

- [ ] Enable the repository in Zenodo.
- [ ] Archive the `v1.0.0` GitHub release.
- [ ] Add the Zenodo DOI to `CITATION.cff`, `README.md`, and the manuscript.
- [ ] Replace the placeholders in `MANUSCRIPT_AVAILABILITY_STATEMENT.md`.
- [ ] Preserve the tagged release used for the revised submission.
