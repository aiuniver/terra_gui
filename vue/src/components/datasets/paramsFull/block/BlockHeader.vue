<template>
  <div class="block-header" @drop="onDrop($event)" @dragover.prevent>
    <div v-if="mixinFiles.length" class="block-header__main">
      <Cards>
        <template v-for="(file, i) of mixinFiles">
          <CardFile v-if="file.type === 'folder'" v-bind="file" :key="'files_' + i" @event="event" />
          <CardTable v-if="file.type === 'table'" v-bind="file" :key="'files_' + i" @event="event" />
        </template>
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
import CardTable from '../components/card/CardTable';
import Cards from '../components/card/Cards.vue';
import blockMain from '@/mixins/datasets/blockMain';
export default {
  name: 'BlockHeader',
  components: {
    CardFile,
    CardTable,
    Cards,
  },
  mixins: [blockMain],
  methods: {
    event({ label }) {
      this.mixinFiles = this.mixinFiles.filter(item => item.label !== label);
    },
    onDrop({ dataTransfer }) {
      const data = JSON.parse(dataTransfer.getData('CardDataType'));
      if (this.mixinFiles.length) {
        if (this.mixinFiles.find(item => item.type !== data.type)) {
          this.$Notify.warning({ title: 'Внимание!', message: 'Выбрать можно только одинаковый тип данных' });
          return;
        }
        if (this.mixinFiles.find(item => item.type === 'table')) {
          this.$Notify.warning({ title: 'Внимание!', message: 'Выбрать можно только одину таблицу' });
          return;
        }
      }
      if (!this.mixinFiles.find(item => item.value === data.value)) {
        this.mixinFiles = [...this.mixinFiles, data];
      } else {
        this.$Notify.warning({ title: 'Внимание!', message: 'Каталог уже выбран' });
      }
    },
  },
};
</script>

<style lang="scss" scoped>
.int {
  padding: 10px;
}
.inner {
  padding: 10px;
  height: 100%;
}
.empty {
  width: 10px;
  height: 100%;
}
.block-header {
  width: 100%;
  // height: 100%;
  &__main {
    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;
    height: 100%;
    position: relative;
  }
  &__overlay {
    user-select: none;
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
