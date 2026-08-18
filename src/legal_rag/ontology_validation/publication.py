"""Validate namespace mappings and build a deployable ontology site."""

from __future__ import annotations

import hashlib
import html
import json
import re
import tempfile
from datetime import datetime
from pathlib import Path

from defusedxml import ElementTree
from rdflib import DCTERMS, OWL, RDF, Graph, URIRef

from legal_rag.ontology_validation.publication_config import (
    NamespacePublicationSettings,
    PublicationResource,
)
from legal_rag.ontology_validation.publication_models import (
    NamespacePublicationReport,
    PublicationCheckResult,
    PublishedFileRecord,
)

_CATALOG_NAMESPACE = "urn:oasis:names:tc:entity:xmlns:xml:catalog"


def build_namespace_publication(
    project_root: Path,
    settings: NamespacePublicationSettings,
    output_directory: Path,
    checked_at: datetime,
) -> NamespacePublicationReport:
    """Validate publication inputs and build one fresh static site."""
    checks = (
        _resource_identity_check(settings),
        _catalog_mapping_check(project_root, settings),
        _endpoint_check(settings),
        _license_check(project_root, settings),
        _w3id_bundle_check(settings),
        _output_path_check(output_directory),
    )
    if any(check.status == "failed" for check in checks):
        return NamespacePublicationReport(
            checked_at=checked_at,
            artifact_id=settings.artifact_id,
            artifact_version=settings.artifact_version,
            overall_status="failed",
            checks=checks,
            files=(),
            manifest_path=None,
        )

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_directory.parent) as temporary_directory:
        staging_directory = Path(temporary_directory) / "site"
        files = _write_site(project_root, settings, staging_directory)
        staging_directory.rename(output_directory)
    return NamespacePublicationReport(
        checked_at=checked_at,
        artifact_id=settings.artifact_id,
        artifact_version=settings.artifact_version,
        overall_status="built",
        checks=checks,
        files=files,
        manifest_path=str(output_directory / settings.manifest_file),
    )


def expected_w3id_htaccess(settings: NamespacePublicationSettings) -> str:
    """Return the exact W3ID Apache redirect configuration for the contract."""
    lines = [
        "# /legal-ontology-rag/",
        "# Permanent identifiers for the legal-ontology-rag ontology publication.",
        f"# Maintainer: https://github.com/{settings.github_username}",
        f"# Project: {settings.project_repository}",
        "",
        "Options -MultiViews",
        "AddType text/turtle .ttl",
        "RewriteEngine On",
        "",
    ]
    for resource, path, endpoint_kind in publication_endpoints(settings):
        pattern = re.escape(path)
        target = f"{settings.publication_base_url}{path}"
        lines.extend(
            (
                f"# {resource.resource_id} {endpoint_kind}",
                "RewriteCond %{HTTP_ACCEPT} (text/html|application/xhtml\\+xml) [NC]",
                f"RewriteRule ^{pattern}/?$ {target}/index.html [R={settings.redirect_status},L,NE]",
                "RewriteCond %{HTTP_ACCEPT} (text/turtle|application/x-turtle) [NC]",
                f"RewriteRule ^{pattern}/?$ {target}/ontology.ttl [R={settings.redirect_status},L,NE]",
                f"RewriteRule ^{pattern}/?$ {target}/ontology.ttl [R={settings.redirect_status},L,NE]",
                "",
            )
        )
    lines.extend(
        (
            "# Unknown identifiers remain unresolved instead of falling through to an unrelated resource.",
            "RewriteRule ^ - [R=404,L]",
            "",
        )
    )
    return "\n".join(lines)


def expected_w3id_readme(settings: NamespacePublicationSettings) -> str:
    """Return the exact maintainer and identifier documentation for the W3ID PR."""
    return f"""# legal-ontology-rag

This directory defines permanent identifiers for the Vietnamese Road Traffic Legal Ontology and its published modules
and SHACL shapes. Canonical identifiers redirect to the current approved release, while version identifiers remain
bound to immutable publication paths.

The redirects provide HTML landing pages for browser requests and Turtle documents for `text/turtle` requests. The
default representation is Turtle. Redirect targets must be deployed and verified before these rules are submitted.

## Maintainer

- GitHub: [@{settings.github_username}](https://github.com/{settings.github_username})
- Project repository: [{settings.project_repository}]({settings.project_repository})

## License

The ontology publication is licensed under [CC BY 4.0]({settings.license_iri}).
"""


def _resource_identity_check(settings: NamespacePublicationSettings) -> PublicationCheckResult:
    """Require every Turtle source to declare its contracted ontology and version IRIs."""
    issues = []
    for resource in settings.resources:
        graph = Graph().parse(resource.source_path, format="turtle")
        ontology = URIRef(resource.ontology_iri)
        if (ontology, RDF.type, OWL.Ontology) not in graph:
            issues.append(f"missing-ontology-iri:{resource.resource_id}:{resource.ontology_iri}")
        if (ontology, OWL.versionIRI, URIRef(resource.version_iri)) not in graph:
            issues.append(f"missing-version-iri:{resource.resource_id}:{resource.version_iri}")
    return _check("resource-identity", issues)


def _catalog_mapping_check(project_root: Path, settings: NamespacePublicationSettings) -> PublicationCheckResult:
    """Require publication resources to match every canonical and versioned XML catalog mapping."""
    catalog_path = project_root / "ontology" / "catalog-v001.xml"
    root = ElementTree.parse(catalog_path).getroot()
    mappings = {
        item.attrib["name"]: (catalog_path.parent / item.attrib["uri"]).resolve()
        for item in root.findall(f"{{{_CATALOG_NAMESPACE}}}uri")
    }
    issues = []
    for resource in settings.resources:
        expected_path = resource.source_path.resolve()
        issues.extend(
            f"catalog-mismatch:{iri}"
            for iri in (resource.ontology_iri, resource.version_iri)
            if mappings.get(iri) != expected_path
        )
    return _check("catalog-mappings", issues)


def _endpoint_check(settings: NamespacePublicationSettings) -> PublicationCheckResult:
    """Require unique paths whose W3ID identifiers exactly match the contracted IRIs."""
    issues = []
    paths = []
    for resource, path, endpoint_kind in publication_endpoints(settings):
        paths.append(path)
        expected_iri = f"{settings.namespace_root}{path}"
        actual_iri = resource.ontology_iri if endpoint_kind == "canonical" else resource.version_iri
        if actual_iri != expected_iri:
            issues.append(f"endpoint-iri-mismatch:{resource.resource_id}:{endpoint_kind}")
    duplicates = sorted({path for path in paths if paths.count(path) > 1})
    issues.extend(f"duplicate-publication-path:{path}" for path in duplicates)
    return _check("publication-endpoints", issues)


def _license_check(project_root: Path, settings: NamespacePublicationSettings) -> PublicationCheckResult:
    """Require the scoped license file and matching root ontology metadata."""
    root_resource = settings.resources[0]
    graph = Graph().parse(root_resource.source_path, format="turtle")
    has_license = (URIRef(root_resource.ontology_iri), DCTERMS.license, URIRef(settings.license_iri)) in graph
    license_path = project_root / "ontology" / "LICENSE"
    issues = []
    if not license_path.is_file() or license_path.stat().st_size == 0:
        issues.append("missing-ontology-license-file")
    if not has_license:
        issues.append("ontology-license-metadata-mismatch")
    return _check("publication-license", issues)


def _w3id_bundle_check(settings: NamespacePublicationSettings) -> PublicationCheckResult:
    """Require the committed W3ID PR bundle to equal the contract-derived files."""
    issues = []
    if settings.htaccess_path.read_text(encoding="utf-8") != expected_w3id_htaccess(settings):
        issues.append("w3id-htaccess-out-of-date")
    if settings.readme_path.read_text(encoding="utf-8") != expected_w3id_readme(settings):
        issues.append("w3id-readme-out-of-date")
    return _check("w3id-pr-bundle", issues)


def _output_path_check(output_directory: Path) -> PublicationCheckResult:
    """Require a fresh deployment directory so stale files cannot survive a build."""
    issues = (f"publication-output-exists:{output_directory}",) if output_directory.exists() else ()
    return _check("fresh-output-path", issues)


def _write_site(
    project_root: Path,
    settings: NamespacePublicationSettings,
    output_directory: Path,
) -> tuple[PublishedFileRecord, ...]:
    """Write HTML and byte-identical Turtle representations plus the site manifest."""
    output_directory.mkdir(parents=True)
    for resource, path, endpoint_kind in publication_endpoints(settings):
        endpoint_directory = output_directory / Path(path)
        endpoint_directory.mkdir(parents=True)
        (endpoint_directory / "ontology.ttl").write_bytes(resource.source_path.read_bytes())
        (endpoint_directory / "index.html").write_text(
            landing_page_content(settings, resource, path, endpoint_kind),
            encoding="utf-8",
            newline="\n",
        )
    files = _site_file_records(output_directory)
    manifest = {
        "schema_version": "0.1.0",
        "artifact_id": settings.artifact_id,
        "artifact_version": settings.artifact_version,
        "namespace_root": settings.namespace_root,
        "publication_base_url": settings.publication_base_url,
        "redirect_status": settings.redirect_status,
        "representations": {"html": "text/html; charset=utf-8", "turtle": "text/turtle; charset=utf-8"},
        "endpoints": [
            {
                "resource_id": resource.resource_id,
                "kind": endpoint_kind,
                "iri": f"{settings.namespace_root}{path}",
                "html_url": f"{settings.publication_base_url}{path}/index.html",
                "turtle_url": f"{settings.publication_base_url}{path}/ontology.ttl",
                "source_file": resource.source_path.resolve().relative_to(project_root.resolve()).as_posix(),
                "source_sha256": _sha256(resource.source_path.read_bytes()),
            }
            for resource, path, endpoint_kind in publication_endpoints(settings)
        ],
        "files": [item.to_dict() for item in files],
    }
    manifest_path = output_directory / settings.manifest_file
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return _site_file_records(output_directory)


def landing_page_content(
    settings: NamespacePublicationSettings,
    resource: PublicationResource,
    path: str,
    endpoint_kind: str,
) -> str:
    """Return one self-contained human-readable ontology landing page."""
    title = html.escape(resource.label)
    iri = resource.ontology_iri if endpoint_kind == "canonical" else resource.version_iri
    canonical_link = f"{settings.namespace_root}{resource.canonical_path}"
    version_link = f"{settings.namespace_root}{resource.version_path}"
    turtle_url = f"{settings.publication_base_url}{path}/ontology.ttl"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: light; --ink: #17211b; --paper: #f5f0e4; --accent: #b73b25; --line: #c7bda7; }}
    body {{ margin: 0; color: var(--ink); background: linear-gradient(135deg, #efe4cc, var(--paper) 48%, #dce8df); font: 18px/1.6 Georgia, serif; }}
    main {{ max-width: 880px; margin: 8vh auto; padding: 3rem; background: rgba(255,255,255,.72); border: 1px solid var(--line); box-shadow: 12px 12px 0 #21382b; }}
    h1 {{ margin-top: 0; font-size: clamp(2.3rem, 7vw, 5rem); line-height: .98; letter-spacing: -.04em; }}
    .eyebrow {{ color: var(--accent); font: 700 .8rem/1.2 ui-monospace, monospace; letter-spacing: .14em; text-transform: uppercase; }}
    dl {{ display: grid; grid-template-columns: 9rem 1fr; gap: .6rem 1rem; border-top: 1px solid var(--line); padding-top: 1.5rem; }}
    dt {{ font-weight: 700; }} dd {{ margin: 0; overflow-wrap: anywhere; }}
    a {{ color: #8e2d1d; text-underline-offset: .18em; }}
    @media (max-width: 640px) {{ main {{ margin: 0; padding: 2rem 1.2rem; box-shadow: none; }} dl {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <p class="eyebrow">Ontology publication · {html.escape(endpoint_kind)}</p>
    <h1>{title}</h1>
    <p>This page documents a permanent identifier in the legal-ontology-rag namespace. Machine-readable Turtle is
    available as an exact copy of the reviewed repository source.</p>
    <dl>
      <dt>Identifier</dt><dd><a href="{html.escape(iri)}">{html.escape(iri)}</a></dd>
      <dt>Canonical IRI</dt><dd><a href="{canonical_link}">{canonical_link}</a></dd>
      <dt>Version IRI</dt><dd><a href="{version_link}">{version_link}</a></dd>
      <dt>Representation</dt><dd><a href="{turtle_url}">Turtle source</a></dd>
      <dt>License</dt><dd><a href="{settings.license_iri}">CC BY 4.0</a></dd>
      <dt>Project</dt><dd><a href="{settings.project_repository}">{settings.project_repository}</a></dd>
    </dl>
  </main>
</body>
</html>
"""


def publication_endpoints(
    settings: NamespacePublicationSettings,
) -> tuple[tuple[PublicationResource, str, str], ...]:
    """Return canonical and versioned endpoints in stable contract order."""
    return tuple(
        endpoint
        for resource in settings.resources
        for endpoint in (
            (resource, resource.canonical_path, "canonical"),
            (resource, resource.version_path, "version"),
        )
    )


def _site_file_records(output_directory: Path) -> tuple[PublishedFileRecord, ...]:
    """Hash all generated site files in path order."""
    return tuple(
        _published_file(output_directory, path)
        for path in sorted(item for item in output_directory.rglob("*") if item.is_file())
    )


def _published_file(output_directory: Path, path: Path) -> PublishedFileRecord:
    """Return one generated file record with its expected media type."""
    media_types = {
        ".html": "text/html; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".ttl": "text/turtle; charset=utf-8",
    }
    content = path.read_bytes()
    return PublishedFileRecord(
        path=path.relative_to(output_directory).as_posix(),
        media_type=media_types[path.suffix],
        sha256=_sha256(content),
        size_bytes=len(content),
    )


def _check(check_id: str, issues: list[str] | tuple[str, ...]) -> PublicationCheckResult:
    """Build one stable namespace-publication check result."""
    issues = tuple(sorted(issues))
    return PublicationCheckResult(check_id, "failed" if issues else "passed", issues)


def _sha256(content: bytes) -> str:
    """Return the lowercase SHA-256 digest for bytes."""
    return hashlib.sha256(content).hexdigest()
