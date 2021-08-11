<template>
  <div class="block-header" @drop="onDrop($event)" @dragover.prevent>
    <div v-if="filesDrop.length" class="block-header__main">
      <Cards>
        <template v-for="(file, i) of filesDrop">
          <CardFile v-if="file.type === 'folder'" v-bind="file" :key="'files_' + i" />
        </template>
        <!-- <CardTable/> -->
      </Cards>
      <div class="empty"></div>
    </div>
    <div v-else class="inner">
      <div class="block-header__overlay">
        <div class="block-header__overlay--icon"></div>
        <div class="block-header__overlay--title">Перетащите папку или файл для начала работы с содержимым архива</div>
      </div>
    </div>
  </div>
</template>

<script>
import CardFile from '../components/card/CardFile.vue';
// import CardTable from "../components/card/CardTable";
import Cards from '../components/card/Cards.vue';
export default {
  name: 'BlockHeader',
  components: {
    CardFile,
    // CardTable,
    Cards,
  },
  data: () => ({
    
  }),
  computed: {
    filesDrop: {
      set(value) {
        this.$store.dispatch('datasets/setFilesDrop', value);
      },
      get() {
        return this.$store.getters['datasets/getFilesDrop'];
      },
    },
  },
  methods: {
    onDrop({ dataTransfer }) {
      const data = JSON.parse(dataTransfer.getData('CardDataType'));
      const index = this.filesDrop.findIndex(({ label }) => {
        return data.label === label;
      });
      console.log(index);
      if (index === -1) {
        this.filesDrop.push(data);
        this.filesDrop = [...this.filesDrop];
        // console.log(this.filesDrop)
      } else {
        this.$Notify.warning({ title: 'Внимание!', message: 'Каталог уже выбран' });
      }
    },
  },
  mounted() {
    // console.log(this.$el.clientHeight);
  },
};
</script>

<style lang="scss" scoped>
.int {
  padding: 10px;
}
.inner {
  // position: absolute;
  // top: 10px;
  // height: 150px;
  padding: 10px;
  height: 100%;
}
.empty {
  width: 10px;
  height: 100%;
}
.block-header {
  width: 100%;
  height: 100%;
  &__main {
    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;
    height: 100%;
    position: relative;
  }
  &__overlay {
    background: #242f3d;
    border: 1px dashed #2b5278;
    border-radius: 4px;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-self: start;
    &--icon {
      margin-left: 20px;
      display: inline-block;
      background-position: center;
      background-repeat: no-repeat;
      -webkit-user-select: none;
      -moz-user-select: none;
      -ms-user-select: none;
      user-select: none;
      width: 16px;
      height: 13px;
      margin-right: 20px;
      background-image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNiAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTggM1YxLjQxQzggMC41MjAwMDIgOS4wOCAwLjA3MDAwMjIgOS43MSAwLjcwMDAwMkwxNS4zIDYuMjlDMTUuNjkgNi42OCAxNS42OSA3LjMxIDE1LjMgNy43TDkuNzEgMTMuMjlDOS4wOCAxMy45MiA4IDEzLjQ4IDggMTIuNTlWMTFIMUMwLjQ1IDExIDAgMTAuNTUgMCAxMFY0QzAgMy40NSAwLjQ1IDMgMSAzSDhaIiBmaWxsPSIjQTdCRUQzIi8+Cjwvc3ZnPgo=);
    }
    &--title {
      width: 280px;
      font-family: Open Sans;
      font-style: normal;
      font-weight: normal;
      font-size: 14px;
      line-height: 24px;
    }
  }
}
</style>