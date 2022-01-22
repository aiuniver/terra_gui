<template>
  <div class="params">
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
</template>

<script>
import DatasetPreview from '@/components/datasets/components/create/DatasetPreview';
import DatasetSettings from '@/components/datasets/components/create/DatasetSettings';
import DatasetDownloadTabs from '@/components/datasets/components/create/DatasetDownloadTabs';
import DatasetHelpers from '@/components/datasets/components/create/DatasetHelpers';
export default {
  components: {
    DatasetPreview,
    DatasetSettings,
    DatasetDownloadTabs,
    DatasetHelpers,
  },
  data: () => ({
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
};
</script>

<style lang="scss">
.params {
    height: 100%;
}
</style>