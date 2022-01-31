<template>
  <div class="params">
    <div class="params__body">
      <div class="params__header">Данные</div>
      <scrollbar>
        <div class="params__inner">
          <component :is="getComp.component" :list="preview" />
          <!-- <Download class="mt-4" />

          <Helpers />

          <FileManager @chooseFile="chooseFile" :list="files" />

          <Preview @choosePreview="choosePreview" :list="preview" />

          <Settings /> -->
        </div>
      </scrollbar>
    </div>
    <div class="params__footer">
      <Pagination v-model="value" :title="getComp.title" />
    </div>
  </div>
</template>

<script>
import Preview from './Preview';
import Settings from './Settings';
import Download from './Download';
import Helpers from './Helpers';
import Pagination from './Pagination';
export default {
  components: {
    Preview,
    Settings,
    Download,
    Pagination,
    Helpers,
  },
  data: () => ({
    value: 1,
    list: [
      { id: 1, title: 'Download', component: 'download' },
      { id: 2, title: 'Preview', component: 'Preview' },
      { id: 3, title: 'Settings', component: 'settings' },
      { id: 4, title: 'Helpers', component: 'helpers' },
    ],

    preview: [
      {
        id: 1,
        path: 'http://u01.appmifile.com/images/2017/04/15/c9b37bf4-2bed-466f-b2a1-e3bee2e54d7c.jpg',
        label: 'Мерседес 1',
      },
      {
        id: 2,
        path: 'https://i.pinimg.com/originals/b3/cf/82/b3cf8221bf35baf3d4faa68473811fc9.jpg',
        label: 'Мерседес 2',
      },
      {
        id: 3,
        path: 'https://storge.pic2.me/c/1360x800/510/029.jpg',
        label: 'Мерседес 3',
      },
    ],
    files: [
      {
        id: 1,
        label: 'Мерседес',
        list: [
          {
            label: 'model_1_10_06_mnistasd',
          },
          {
            label: 'model_asda_06_mnisasdt',
          },
          {
            label: 'moasdl_asdasdasda_06_mnisasdtasd',
          },
        ],
      },
      {
        id: 2,
        label: 'Рено',
        list: [
          {
            label: 'model_1_10_06_mnist23d23',
          },
        ],
      },
      {
        id: 3,
        label: 'Рено',
        list: [
          {
            label: 'model_1_10_06_mnistd23dsae',
          },
          {
            label: 'model_1_10_06_mnistfe24c',
          },
          {
            label: 'model_1_10_06_mnistbg5rgvt',
          },
          {
            label: 'model_1_10_06_mnisttbyg3v5',
          },
          {
            label: 'model_1_10_06_mnistn5h4e',
          },
        ],
      },
    ],
  }),
  computed: {
    getComp() {
      return this.list.find(i => i.id === this.value);
    },
  },
  methods: {
    chooseFile(item) {
      console.log(item);
    },
    choosePreview(id) {
      console.log(id);
    },
    handleWorkspaceAction(action) {
      console.log(action);
    },
  },
};
</script>

<style lang="scss">
.params {
  position: relative;
  display: flex;
  flex-direction: column;
  height: 100%;
  &__header {
    height: 50px;
    display: flex;
    align-items: center;
    border-bottom: 1px solid black;
    padding: 0 20px;
  }
  &__inner {
    padding: 0 20px;
    height: 100%;
  }
  &__body {
    flex: 1 1 auto;
    overflow: hidden;
  }
  &__footer {
    border-top: 1px solid black;
    flex: 0 0 70px;
    padding: 10px 20px;
  }
}
</style>