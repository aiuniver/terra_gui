<template>
  <div class="tabs-download">
    <div class="tabs-download-list flex align-center">
      <div
        :class="['tabs-download-list__item', { 'tabs-download-list__item--active': +active === +tab }]"
        v-for="{ text, tab } in items"
        :key="JSON.stringify(text + tab)"
        @click="active = tab"
      >
        <p>{{ text }}</p>
      </div>
    </div>
    <div class="tabs-download-content mt-10">
      <TField icon="google-drive" label="Выберите файл на Google диске" v-if="active === 0">
        <DSelect icon="google-drive" v-model="select" placeholder="Введите имя файла" :list="list" />
      </TField>
      <TField label="Загрузите по ссылке" v-if="active === 1">
        <DInputText v-model="url" placeholder="URL" />
      </TField>
    </div>
  </div>
</template>

<script>
import TField from '@/components/global/forms/TField';
import DSelect from '@/components/global/forms/DSelect';
import DInputText from '@/components/global/forms/DInputText';

export default {
  components: { TField, DSelect, DInputText },
  name: 'DatasetDownloadTabs',
  data: () => ({
    url: '',
    select: '',
    list: [
      {
        label: 'Селект 1',
        value: 1,
      },
      {
        label: 'Селект 2',
        value: 2,
      },
      {
        label: 'Селект 3',
        value: 3,
      },
      {
        label: 'Селект 4',
        value: 4,
      },
    ],
    items: [
      { text: 'Google диск', tab: 0 },
      { text: 'URL', tab: 1 },
    ],
    active: 0,
  }),
};
</script>

<style lang="scss" scoped>
.tabs-download {
  &-list {
    &__item {
      width: 50%;
      text-align: center;
      position: relative;
      cursor: pointer;
      color: $color-light-gray;
      transition: color 0.3s ease;
      &:hover {
        color: $color-light-blue;
      }
      &--active {
        &::before {
          content: '';
          width: 235px;
          height: 1px;
          position: absolute;
          bottom: 0;
          left: 50%;
          background: $color-light-blue;
          transform: translateX(-50%);
        }
      }
    }
  }
}
</style>
