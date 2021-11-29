<template>
  <main class="page page-datasets">
    <div class="page-datasets-inner">
      <div class="page-datasets-inner__workspace"><router-view></router-view></div>
      <BasePanel @action="handleActionPanel">
        <template v-if="component === 'DatasetTabsDownload'">
          <BasePanelContent noMargin>
            <template #header>Данные</template>
            <template #content>
              <DatasetDownloadTabs class="mt-4" />
            </template>
          </BasePanelContent>
        </template>
        <template v-if="component === 'DatasetHelpers'">
          <BasePanelContent>
            <template #header>Данные</template>
            <template #content>
              <DatasetHelpers />
            </template>
          </BasePanelContent>
        </template>
        <template v-if="component === 'DataEnterDataset'">
          <BasePanelContent>
            <template #header>Данные</template>
            <template #content>
              <div class="mb-4">Выберите папку/файл</div>
              <FileManager @chooseFile="chooseFile" :list="list" />
            </template>
          </BasePanelContent>
          <BasePanelContent>
            <template #header>Предпросмотр</template>
            <template #content>
              <DatasetPreview @choosePreview="choosePreview" :list="preview" />
            </template>
          </BasePanelContent>
          <BasePanelContent>
            <template #header>Настройки</template>
            <template #content>
              <DatasetSettings />
            </template>
          </BasePanelContent>
        </template>
      </BasePanel>
    </div>
    <Layer v-bind="layerData1" style="position: absolute; top: 150px; left: 50%" />
    <Layer v-bind="layerData2" style="position: absolute; top: 150px; left: 30%" />
    <Layer v-bind="layerData3" style="position: absolute; top: 150px; left: 10%" />
    <WorkspaceActions @action="handleWorkspaceAction" />
  </main>
</template>

<script>
export default {
  components: {
    WorkspaceActions: () => import('@/components/datasets/components/create/WorkspaceActions'),
    DatasetPreview: () => import('@/components/datasets/components/create/DatasetPreview'),
    DatasetSettings: () => import('@/components/datasets/components/create/DatasetSettings'),
    DatasetDownloadTabs: () => import('@/components/datasets/components/create/DatasetDownloadTabs'),
    DatasetHelpers: () => import('@/components/datasets/components/create/DatasetHelpers'),
    Layer: () => import('@/components/datasets/components/Layer'),
  },

  name: 'Datasets',
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
    handleActionPanel(action) {
      let idx = this.allComponents.findIndex(el => el === this.component);
      if (action === 'next') {
        if (+idx < this.allComponents.length - 1) {
          this.component = this.allComponents[++idx];
        } else {
          this.component = this.allComponents[0];
        }
      } else if (action === 'prev') {
        if (+idx !== 0) {
          this.component = this.allComponents[--idx];
        } else {
          this.component = this.allComponents[this.allComponents.length - 1];
        }
      }
    },
  },
  data: () => ({
    layerData1: {
      type: 'input',
      title: 'input_Вход 1: ',
      data: [22, 28, 1],
      error: '',
    },
    layerData2: {
      type: 'middle',
      title: 'input_Вход 1: ',
      data: [22, 28, 1],
      error: 'asdsad',
    },
    layerData3: {
      type: 'output',
      title: 'input_Вход 1: ',
      data: [22, 28, 1],
      error: '',
    },
    component: 'DatasetHelpers',
    allComponents: ['DatasetTabsDownload', 'DatasetHelpers', 'DataEnterDataset'],
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
    list: [
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
};
</script>

<style lang="scss" scoped>
.page-datasets {
  position: relative;

  &-inner::v-deep {
    display: flex;
    height: 100%;
    justify-content: space-between;
  }
}
</style>