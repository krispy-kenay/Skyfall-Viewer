export function decodeHeader(plyArrayBuffer: ArrayBuffer): [number, Record<string, string>, DataView] {
    /* decodes the .ply file header and returns a tuple of:
        * - vertexCount: number of vertices in the point cloud
        * - propertyTypes: a map from property names to their types
        * - vertexData: a DataView of the vertex data
    */

    const decoder = new TextDecoder();
    let headerOffset = 0;
    let headerText = '';

    while (true) {
        const headerChunk = new Uint8Array(plyArrayBuffer, headerOffset, 50);
        headerText += decoder.decode(headerChunk);
        headerOffset += 50;

        if (headerText.includes('end_header')) {
            break;
        }
    }

    const headerLines = headerText.split('\n');

    let vertexCount = 0;
    let propertyTypes: Record<string, string> = {};

    for (let i = 0; i < headerLines.length; i++) {
        const line = headerLines[i].trim();
        if (line.startsWith('element vertex')) {
            const vertexCountMatch = line.match(/\d+/);
            if (vertexCountMatch) {
                vertexCount = parseInt(vertexCountMatch[0]);
            }
        } else if (line.startsWith('property')) {
            const propertyMatch = line.match(/(\w+)\s+(\w+)\s+(\w+)/);
            if (propertyMatch) {
                const propertyType = propertyMatch[2];
                const propertyName = propertyMatch[3];
                propertyTypes[propertyName] = propertyType;
            }
        } else if (line === 'end_header') {
            break;
        }
    }

    const vertexByteOffset = headerText.indexOf('end_header') + 'end_header'.length + 1;
    const vertexData = new DataView(plyArrayBuffer, vertexByteOffset);

    return [
        vertexCount,
        propertyTypes,
        vertexData,
    ];
}

export function readRawVertex(offset: number, vertexData: DataView, propertyTypes: Record<string, string>): [number, Record<string, number>] {
    /* reads a single vertex from the vertexData DataView and returns a tuple of:
        * - offset: the offset of the next vertex in the vertexData DataView
        * - rawVertex: a map from property names to their values
    */
    let rawVertex: Record<string, number> = {};

    for (const property in propertyTypes) {
        const propertyType = propertyTypes[property];
        if (propertyType === 'float') {
            rawVertex[property] = vertexData.getFloat32(offset, true);
            offset += Float32Array.BYTES_PER_ELEMENT;
        } else if (propertyType === 'uchar') {
            rawVertex[property] = vertexData.getUint8(offset) / 255.0;
            offset += Uint8Array.BYTES_PER_ELEMENT;
        }
    }

    return [offset, rawVertex];
}

export function nShCoeffs(sphericalHarmonicsDegree: number): number {
    /* returns the expected number of spherical harmonics coefficients */
    if (sphericalHarmonicsDegree === 0) {
        return 1;
    } else if (sphericalHarmonicsDegree === 1) {
        return 4;
    } else if (sphericalHarmonicsDegree === 2) {
        return 9;
    } else if (sphericalHarmonicsDegree === 3) {
        return 16;
    } else {
        throw new Error(`Unsupported SH degree: ${sphericalHarmonicsDegree}`);
    }
}