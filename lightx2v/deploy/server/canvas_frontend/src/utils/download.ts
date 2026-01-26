import { DataType } from '../../types';

/**
 * 下载文件
 */
export const downloadFile = (
  content: string,
  fileName: string,
  type: DataType
): void => {
  const link = document.createElement('a');

  if (type === DataType.TEXT) {
    const contentString = typeof content === 'object'
      ? JSON.stringify(content, null, 2)
      : content;
    const blob = new Blob([contentString], { type: 'text/plain' });
    link.href = URL.createObjectURL(blob);
  } else {
    link.href = content;
  }

  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};
