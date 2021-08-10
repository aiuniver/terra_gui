<template>
  <div class="block-header" @drop="onDrop($event)" @dragover.prevent>
    <div v-if="files.length" class="block-header__main">
      <Cards>
          <template v-for="({ title, color, type }, i) of files">
            <CardFile
              v-if="type === 'folder'"
              :title="title"
              :color="color"
              :type="type"
              :key="'files_' + i"
            />

          </template>
          <!-- <CardTable/> -->
      </Cards>
      <div class="empty"></div>
    </div>
    <div v-else class="inner">
      <div class="block-header__overlay">
        <div class="block-header__overlay--icon"></div>
        <div class="block-header__overlay--title">
          Перетащите папку или файл для начала работы с содержимым архива
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import CardFile from "../components/card/CardFile.vue";
// import CardTable from "../components/card/CardTable";
import Cards from "../components/card/Cards.vue";
export default {
  name: "BlockHeader",
  components: {
    CardFile,
    // CardTable,
    Cards,
  },
  data: () => ({
    files: [
      // { title: 'BMW', color: '#FFB054'},
      // { title: 'AUDI', color: '#8E51F2'},
      // { title: 'Ferrari', color: '#89D764'}
    ],
  }),
  methods: {
    onDrop(e) {
      console.log(e);
      // this.files.push({ title: '', color: ''})
      const data = e.dataTransfer.getData("CardDataType");
      const { title, color, type } = JSON.parse(data);
      console.log(JSON.parse(data));
      this.files.push({ title, color, type });
    },
  },
  mounted() {
    // console.log(this.$el.clientHeight);
  },
};
</script>

<style lang="scss" scoped>
.int{
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