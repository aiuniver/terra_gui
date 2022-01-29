<template>
  <div class="d-upload">
    <div class="d-upload-label mb-3">
      <p>{{ label }}</p>
    </div>
    <div
      v-show="Object.keys(file).length === 0 && !behindFile.length"
      class="d-upload-container"
      ref="d-upload-container"
      :style="{ border: `1px solid ${error ? '#CA5035' : '#65B9F4'}` }"
    >
      <div v-for="line in 4" :key="'line-' + line" class="d-upload-container-line"></div>
      <div class="d-upload-error" v-if="error.length">
        <p>{{ error }}</p>
      </div>
      <input v-show="false" ref="file" type="file" @change="onInputFileChange" />
      <label for="d-file-input">asd</label>
      <p class="d-upload-text mb-5">Загрузите файл простым переносом или по кнопке ниже</p>
      <d-button style="width: 100%" color="secondary" direction="left" @click="$refs.file.click()">
        Загрузить файл
      </d-button>
    </div>

    <div class="d-upload-container-image" v-show="Object.keys(file).length || behindFile.length">
      <img :src="behindFile.length ? behindFile : '#'" class="d-upload-image" />
      <div @click="removeFile" class="d-upload-image--remove">
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path
            d="M8.59 0L5 3.59L1.41 0L0 1.41L3.59 5L0 8.59L1.41 10L5 6.41L8.59 10L10 8.59L6.41 5L10 1.41L8.59 0Z"
            fill="#A7BED3"
          />
        </svg>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'd-upload',
  data: () => ({
    dragEnter: false,
    uploading: false,
    file: {},
    error: '',
  }),
  components: {},
  props: {
    label: { type: String, default: 'Обложка проекта' },
    behindFile: { type: String, default: '' },
  },
  methods: {
    removeFile() {
      this.file = {};
      this.uploading = false;
      this.$emit('removeFile');
    },
    preventAndStop(ev) {
      ev.preventDefault();
      ev.stopPropagation();
    },
    onDragEnter(ev) {
      if (ev.dataTransfer?.types?.[0] === 'Files') this.dragEnter = true;
    },
    onDrop(ev) {
      if (this.dragEnter) this.initUpload(ev.dataTransfer.files);
    },
    onInputFileChange(ev) {
      this.initUpload(ev.target.files);
    },
    checkError(file) {
      if (file.length === 0) return true;
      const types = ['jpeg', 'png', 'jpg'];
      const type = file[0].type.split('/')[1];
      if (!types.includes(type)) {
        this.error = 'Неверный формат файла';
        return true;
      }
      return false;
    },
    initUpload(file) {
      this.error = '';
      if (!this.checkError(file)) {
        this.dragEnter = false;
        this.file = file;
        this.readerImage(file);
      }
    },
    readerImage(file) {
      const reader = new FileReader();
      const image = document.querySelector('.d-upload-image');
      reader.onload = function (e) {
        image.src = e.target.result;
      };
      reader.readAsDataURL(file[0]);
      this.uploading = true;
      this.$emit('uploadFile', { file: file[0].name });
    },
  },
  mounted() {
    const arena = this.$refs['d-upload-container'];
    ['dragenter', 'dragover', 'drop'].forEach(evName => {
      arena.addEventListener(evName, this.preventAndStop);
    });
    arena.addEventListener('dragenter', this.onDragEnter);
    arena.addEventListener('drop', this.onDrop);
  },
};
</script>

<style lang="scss" scoped>
.d-upload {
  &-error {
    position: absolute;
    top: 5px;
    left: 5px;
    font-family: Open Sans;
    background: #ca5035;
    opacity: 0.9;
    box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.25);
    border-radius: 4px;
    padding: 5px 10px;
    p {
      font-style: normal;
      font-weight: normal;
      font-size: 10px;
      color: #ffffff;
    }
  }
  &-label {
    p {
      font-family: 'Open Sans';
      font-style: normal;
      font-weight: normal;
      font-size: 12px;
      color: #a7bed3;
    }
  }
  &-image {
    &--remove {
      position: absolute;
      right: 10px;
      display: flex;
      top: 10px;
      justify-content: center;
      align-items: center;
      right: 10px;
      height: 24px;
      width: 24px;
      background: rgba(36, 47, 61, 0.5);
      border-radius: 4px;
      cursor: pointer;
      transition: background 0.3s ease;
      &:hover {
        background: rgba(36, 47, 61, 1);
      }
    }
  }
  &-container {
    position: relative;
    padding: 0px 40px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    max-width: 341px;
    height: 230px;
    background: rgba(36, 47, 61, 0.5);
    &-line {
      width: 34%;
      z-index: 10;
      position: absolute;
      height: 1px;
      background: #1d232b;
    }
    &-line:nth-child(1) {
      top: -1px;
    }
    &-line:nth-child(2) {
      top: 50%;
      left: -57px;
      transform: translateY(-50%) rotate(90deg);
    }
    &-line:nth-child(3) {
      top: 50%;
      right: -58px;
      transform: translateY(-50%) rotate(90deg);
    }
    &-line:nth-child(4) {
      bottom: -1px;
    }
    &-image {
      position: relative;
      max-width: 341px;
      height: 230px;
      background: rgba(36, 47, 61, 0.5);
      img {
        object-fit: cover;
        width: 100%;
        height: 100%;
      }
    }
  }
  &-text {
    text-align: center;
    font-family: 'Segoe UI';
    font-style: normal;
    font-weight: normal;
    font-size: 14px;
    color: rgba(246, 251, 253, 0.6);
  }
}
</style>