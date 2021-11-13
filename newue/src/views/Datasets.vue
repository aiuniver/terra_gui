<template>
  <main class="page page-datasets">
    <div class="page-datasets-inner">
      <div class="page-datasets-inner__workspace">asd</div>
      <BasePanel>
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
            <Preview @choosePreview="choosePreview" :list="preview" />
          </template>
        </BasePanelContent>
        <BasePanelContent>
          <template #header>Настройки</template>
          <template #content>
            <Settings />
          </template>
        </BasePanelContent>
      </BasePanel>
    </div>
    <WorkspaceActions @action="handleWorkspaceAction" />
  </main>
</template>

<script>
import BasePanel from '@/components/datasets/components/panel/BasePanel';
import BasePanelContent from '@/components/datasets/components/panel/BasePanelContent';

import FileManager from '@/components/datasets/components/FileManager';
import Preview from '@/components/datasets/components/Preview';
import Settings from '@/components/datasets/components/Settings';
import WorkspaceActions from '@/components/datasets/components/WorkspaceActions';

export default {
  components: {
    BasePanelContent,
    BasePanel,
    Preview,
    FileManager,
    Settings,
    WorkspaceActions,
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
  },
  data: () => ({
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