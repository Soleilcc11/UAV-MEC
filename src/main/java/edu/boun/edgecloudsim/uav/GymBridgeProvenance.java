package edu.boun.edgecloudsim.uav;

import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import org.json.JSONArray;
import org.json.JSONObject;

/** Content-addressed identity for the running Java service and its XML inputs. */
final class GymBridgeProvenance {
    private GymBridgeProvenance() { }

    static JSONObject describe(String settingsPath, String edgeDevicesPath,
            String applicationsPath) {
        try {
            Path codeSource = codeSource();
            Path repository = findRepositoryRoot(codeSource);
            JSONArray files = configFiles(
                    Paths.get(settingsPath),
                    Paths.get(edgeDevicesPath),
                    Paths.get(applicationsPath));
            JSONObject runtime = new JSONObject()
                    .put("artifact_sha256", hashArtifact(codeSource))
                    .put("classes_current", classesCurrent(repository, codeSource));
            if (repository != null) {
                runtime.put("git_commit_sha", readGitCommit(repository))
                        .put("source_tree_sha256", hashSourceTree(repository));
            }
            return new JSONObject()
                    .put("environment", new JSONObject().put("files", files))
                    .put("runtime", runtime);
        } catch (Exception exception) {
            throw new IllegalStateException(
                    "Cannot establish GymBridge runtime provenance", exception);
        }
    }

    private static Path codeSource() throws Exception {
        URI location = GymBridgeSession.class.getProtectionDomain()
                .getCodeSource().getLocation().toURI();
        return Paths.get(location).toAbsolutePath().normalize();
    }

    private static JSONArray configFiles(Path... paths) throws IOException {
        List<Path> sorted = new ArrayList<>();
        for (Path path : paths) {
            Path normalized = path.toAbsolutePath().normalize();
            if (!Files.isRegularFile(normalized)) {
                throw new IOException("Environment config is not a file: " + normalized);
            }
            sorted.add(normalized);
        }
        sorted.sort(Comparator.comparing(path -> path.getFileName().toString()));
        JSONArray files = new JSONArray();
        String previousName = null;
        for (Path path : sorted) {
            String name = path.getFileName().toString();
            if (name.equals(previousName)) {
                throw new IOException("Duplicate environment config filename: " + name);
            }
            previousName = name;
            files.put(new JSONObject()
                    .put("name", name)
                    .put("sha256", hashFile(path)));
        }
        return files;
    }

    private static Path findRepositoryRoot(Path codeSource) {
        Path start = Files.isDirectory(codeSource) ? codeSource : codeSource.getParent();
        for (Path candidate = start; candidate != null; candidate = candidate.getParent()) {
            if (Files.exists(candidate.resolve(".git"))
                    && Files.isDirectory(candidate.resolve("src/main/java"))) {
                return candidate;
            }
        }
        Path current = Paths.get("").toAbsolutePath().normalize();
        for (Path candidate = current; candidate != null; candidate = candidate.getParent()) {
            if (Files.exists(candidate.resolve(".git"))
                    && Files.isDirectory(candidate.resolve("src/main/java"))) {
                return candidate;
            }
        }
        return null;
    }

    private static String readGitCommit(Path repository) throws IOException {
        Path git = repository.resolve(".git");
        Path gitDirectory = git;
        if (Files.isRegularFile(git)) {
            String pointer = readUtf8(git).trim();
            if (!pointer.startsWith("gitdir:")) {
                throw new IOException("Unsupported .git pointer");
            }
            gitDirectory = repository.resolve(pointer.substring("gitdir:".length()).trim())
                    .normalize();
        }
        String head = readUtf8(gitDirectory.resolve("HEAD")).trim();
        if (!head.startsWith("ref:")) {
            return head;
        }
        String reference = head.substring("ref:".length()).trim();
        Path looseReference = gitDirectory.resolve(reference);
        if (Files.isRegularFile(looseReference)) {
            return readUtf8(looseReference).trim();
        }
        Path packedRefs = gitDirectory.resolve("packed-refs");
        if (Files.isRegularFile(packedRefs)) {
            for (String line : Files.readAllLines(packedRefs, StandardCharsets.UTF_8)) {
                if (!line.startsWith("#") && !line.startsWith("^")
                        && line.endsWith(" " + reference)) {
                    return line.substring(0, line.indexOf(' '));
                }
            }
        }
        throw new IOException("Cannot resolve Git HEAD reference: " + reference);
    }

    private static String hashArtifact(Path codeSource) throws IOException {
        if (Files.isRegularFile(codeSource)) {
            return hashFile(codeSource);
        }
        List<Path> files = listRegularFiles(codeSource, path ->
                path.getFileName().toString().endsWith(".class"));
        if (files.isEmpty()) {
            throw new IOException("No runtime classes found under " + codeSource);
        }
        return hashTree(codeSource, files);
    }

    private static String hashSourceTree(Path repository) throws IOException {
        List<Path> files = new ArrayList<>();
        Path pom = repository.resolve("pom.xml");
        if (Files.isRegularFile(pom)) files.add(pom);
        files.addAll(listRegularFiles(repository.resolve("src/main/java"), path -> true));
        files.addAll(listRegularFiles(repository.resolve("src/main/resources"), path -> true));
        files.sort(Comparator.comparing(path -> repository.relativize(path).toString()));
        return hashTree(repository, files);
    }

    private static boolean classesCurrent(Path repository, Path codeSource) throws IOException {
        if (repository == null) {
            return true;
        }
        List<Path> sources = listRegularFiles(repository.resolve("src/main/java"), path ->
                path.getFileName().toString().endsWith(".java"));
        if (Files.isRegularFile(codeSource)) {
            long artifactTime = Files.getLastModifiedTime(codeSource).toMillis();
            for (Path source : sources) {
                if (Files.getLastModifiedTime(source).toMillis() > artifactTime) return false;
            }
            return true;
        }
        Path sourceRoot = repository.resolve("src/main/java");
        for (Path source : sources) {
            Path relative = sourceRoot.relativize(source);
            String name = relative.getFileName().toString();
            Path classFile = codeSource.resolve(relative).resolveSibling(
                    name.substring(0, name.length() - ".java".length()) + ".class");
            if (!Files.isRegularFile(classFile)
                    || Files.getLastModifiedTime(classFile).toMillis()
                            < Files.getLastModifiedTime(source).toMillis()) {
                return false;
            }
        }
        return true;
    }

    private interface PathPredicate { boolean test(Path path); }

    private static List<Path> listRegularFiles(Path root, PathPredicate predicate)
            throws IOException {
        List<Path> result = new ArrayList<>();
        if (!Files.isDirectory(root)) return result;
        try (java.util.stream.Stream<Path> stream = Files.walk(root)) {
            stream.filter(Files::isRegularFile)
                    .filter(predicate::test)
                    .forEach(result::add);
        }
        result.sort(Comparator.comparing(path -> root.relativize(path).toString()));
        return result;
    }

    private static String hashTree(Path root, List<Path> files) throws IOException {
        MessageDigest digest = sha256();
        for (Path file : files) {
            String relative = root.relativize(file).toString().replace('\\', '/');
            digest.update(relative.getBytes(StandardCharsets.UTF_8));
            digest.update((byte) 0);
            digest.update(Files.readAllBytes(file));
        }
        return hex(digest.digest());
    }

    private static String hashFile(Path path) throws IOException {
        MessageDigest digest = sha256();
        byte[] buffer = new byte[1024 * 1024];
        try (java.io.InputStream input = Files.newInputStream(path)) {
            int count;
            while ((count = input.read(buffer)) >= 0) {
                if (count > 0) digest.update(buffer, 0, count);
            }
        }
        return hex(digest.digest());
    }

    private static String readUtf8(Path path) throws IOException {
        return new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
    }

    private static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 is unavailable", exception);
        }
    }

    private static String hex(byte[] bytes) {
        StringBuilder result = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) result.append(String.format("%02x", value));
        return result.toString();
    }
}
