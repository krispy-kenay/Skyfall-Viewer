import { Float16Array } from '@petamoriken/float16';
import { log, time, timeLog } from './simple-console';
import { decodeHeader, readRawVertex ,nShCoeffs} from './plyreader';
import { C_SIZE_3D_GAUSSIAN, C_SIZE_SH_COEF } from './pointcloud-loader';

export type PointCloud = Awaited<ReturnType<typeof load>>;

export async function load(file: string, device: GPUDevice) {
  const blob = new Blob([file]);
  const arrayBuffer = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = function(event) {
      resolve(event.target.result);  // Resolve the promise with the ArrayBuffer
    };

    reader.onerror = reject;  // Reject the promise in case of an error
    reader.readAsArrayBuffer(blob);
  });

  const [vertexCount, propertyTypes, vertexData] = decodeHeader(arrayBuffer as ArrayBuffer);
  const num_points = vertexCount;

  const hasSH_DC   = 'f_dc_0' in propertyTypes;
  const hasRestSH  = Object.keys(propertyTypes).some(p => p.startsWith('f_rest_'));
  const hasOpacity = 'opacity' in propertyTypes;
  const hasRot     = 'rot_0' in propertyTypes;
  const hasScale   = 'scale_0' in propertyTypes;
  const hasRGB     = 'red' in propertyTypes && 'green' in propertyTypes && 'blue' in propertyTypes;

  const isGaussianFormat = hasSH_DC && hasOpacity && hasRot && hasScale;
  const isSimplePointCloud = hasRGB && !isGaussianFormat;


  let sh_deg = 0;
  let num_coefs = 1;
  const max_num_coefs = 16;
  let shFeatureOrder: string[] = [];

  if (isGaussianFormat) {
    var nRestCoeffs = 0;
    for (const propertyName in propertyTypes) {
      if (propertyName.startsWith('f_rest_')) {
        nRestCoeffs += 1;
      }
    }
    const nCoeffsPerColor = nRestCoeffs / 3;
    sh_deg = Math.sqrt(nCoeffsPerColor + 1) - 1;
    num_coefs = nShCoeffs(sh_deg);
    for (let rgb = 0; rgb < 3; ++rgb) {
      shFeatureOrder.push(`f_dc_${rgb}`);
    }
    for (let i = 0; i < nCoeffsPerColor; ++i) {
      for (let rgb = 0; rgb < 3; ++rgb) {
        shFeatureOrder.push(`f_rest_${rgb * nCoeffsPerColor + i}`);
      }
    }
  } else {
    sh_deg = 0;
    num_coefs = 1;
  }

  log(`num points: ${num_points}`);
  log(`format: ${isGaussianFormat ? 'full 3D Gaussian' : (isSimplePointCloud ? 'simple xyz+rgb' : 'unknown/partial')}`);
  log(`processing loaded attributes...`);
  time();

  // xyz (position), opacity, cov (from rot and scale)
  const gaussian_3d_buffer = device.createBuffer({
    label: 'ply input 3d gaussians data buffer',
    size: num_points * C_SIZE_3D_GAUSSIAN,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const gaussian = new Float16Array(gaussian_3d_buffer.getMappedRange());

  // Spherical harmonic function coeffs
  const sh_buffer = device.createBuffer({
    label: 'ply input 3d gaussians data buffer',
    size: num_points * C_SIZE_SH_COEF,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const sh = new Float16Array(sh_buffer.getMappedRange());

  var readOffset = 0;
  const defaultOpacity = 1.0;
  const defaultSigma = 0.01;
  const defaultScale = Math.log(defaultSigma);
  const defaultRot = [0, 0, 0, 1];

  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;

  for (let i = 0; i < num_points; i++) {
    const [newReadOffset, rawVertex] = readRawVertex(readOffset, vertexData, propertyTypes);
    readOffset = newReadOffset;

    const o = i * (C_SIZE_3D_GAUSSIAN / 2);
    const output_offset = i * max_num_coefs * 3;
    
    if (isGaussianFormat) {
      for (let order = 0; order < num_coefs; ++order) {
        const order_offset = order * 3;
        for (let j = 0; j < 3; ++j) {
          const coeffName = shFeatureOrder[order * 3 + j];
          sh[output_offset +order_offset+j]=rawVertex[coeffName];
        }
      }
    } else if (isSimplePointCloud) {
      const r8 = (rawVertex as any).red ?? 255;
      const g8 = (rawVertex as any).green ?? 255;
      const b8 = (rawVertex as any).blue ?? 255;
      const r = (r8 as number) / 255.0;
      const g = (g8 as number) / 255.0;
      const b = (b8 as number) / 255.0;
      sh[output_offset + 0] = r;
      sh[output_offset + 1] = g;
      sh[output_offset + 2] = b;
      for (let k = 3; k < max_num_coefs * 3; ++k) {
        sh[output_offset + k] = 0.0;
      }
    }

    const x = (rawVertex as any).x;
    const y = (rawVertex as any).y;
    const z = (rawVertex as any).z;
    
    gaussian[o + 0] = x;
    gaussian[o + 1] = y;
    gaussian[o + 2] = z;
    
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
    minZ = Math.min(minZ, z);
    maxZ = Math.max(maxZ, z);

    if (isGaussianFormat && hasOpacity) {
      gaussian[o + 3] = (rawVertex as any).opacity;
    } else {
      gaussian[o + 3] = defaultOpacity;
    }
    if (isGaussianFormat && hasRot) {
      gaussian[o + 4] = (rawVertex as any).rot_0;
      gaussian[o + 5] = (rawVertex as any).rot_1;
      gaussian[o + 6] = (rawVertex as any).rot_2;
      gaussian[o + 7] = (rawVertex as any).rot_3;
    } else {
      gaussian[o + 4] = defaultRot[0];
      gaussian[o + 5] = defaultRot[1];
      gaussian[o + 6] = defaultRot[2];
      gaussian[o + 7] = defaultRot[3];
    }

    if (isGaussianFormat && hasScale) {
      gaussian[o + 8]  = (rawVertex as any).scale_0;
      gaussian[o + 9]  = (rawVertex as any).scale_1;
      gaussian[o + 10] = (rawVertex as any).scale_2;
    } else {
      gaussian[o + 8]  = defaultScale;
      gaussian[o + 9]  = defaultScale;
      gaussian[o + 10] = defaultScale;
    }

  }

  gaussian_3d_buffer.unmap(); 
  sh_buffer.unmap();

  const extentX = maxX - minX;
  const extentY = maxY - minY;
  const extentZ = maxZ - minZ;
  const scene_extent = Math.max(extentX, extentY, extentZ);
  
  const centerX = (minX + maxX) / 2.0;
  const centerY = (minY + maxY) / 2.0;
  const centerZ = (minZ + maxZ) / 2.0;

  timeLog();
  console.log("return result!");
  console.log(`Scene extent: ${scene_extent.toFixed(2)} units`);
  console.log(`Bounding box: [${minX.toFixed(2)}, ${minY.toFixed(2)}, ${minZ.toFixed(2)}] to [${maxX.toFixed(2)}, ${maxY.toFixed(2)}, ${maxZ.toFixed(2)}]`);
  
  return {
    num_points: num_points,
    sh_deg: sh_deg,
    gaussian_3d_buffer,
    sh_buffer,
    scene_extent: scene_extent,
    bounding_box: {
      min: { x: minX, y: minY, z: minZ },
      max: { x: maxX, y: maxY, z: maxZ },
      center: { x: centerX, y: centerY, z: centerZ },
      extent: { x: extentX, y: extentY, z: extentZ },
    },
  };
}
