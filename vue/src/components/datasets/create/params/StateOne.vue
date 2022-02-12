<template>
  <div class="state-one">
    <div class="state-one-list flex align-center">
      <div
        v-for="{ text, tab } in items"
        :class="['state-one-list__item', { 'state-one-list__item--active': isActive(tab) }]"
        :key="`tab_${tab}`"
        @click="onTabs(tab)"
      >
        {{ text }}
      </div>
    </div>
    <div class="state-one-content mt-10">
      <t-field icon="google" label="Выберите файл на Google диске" v-if="project.active === 0">
        <d-auto-complete
          v-model="project.source_path"
          icon="google-drive"
          placeholder="Введите имя файла"
          :list="getFilesSource"
          @click="getDatasetSources"
          @change="onSelect({ mode: 'GoogleDrive', value: $event.value })"
        />
      </t-field>
      <t-field icon="link" label="Загрузите по ссылке" v-if="project.active === 1">
        <d-input-text v-model="project.source_path" placeholder="URL" @blur="onSelect({ mode: 'URL', value: $event.target.value })" />
      </t-field>
    </div>
    <div>
      <t-field label="Название датасета">
        <d-input-text v-model="project.name" />
      </t-field>
      <t-field label="Тип архитектуры">
        <d-input-text v-model="project.task_type" />
      </t-field>
      <div class="mb-2">
        <DTags v-model="project.tags" />
      </div>
    </div>
  </div>
</template>

<script>
import { mapActions, mapGetters } from 'vuex';
import DTags from '@/components/forms/DTags';
export default {
  name: 'DatasetDownloadTabs',
  components: {
    DTags,
  },
  data: () => ({
    items: [
      { text: 'Google диск', tab: 0 },
      { text: 'URL', tab: 1 },
    ],
    active: 0,
  }),
  computed: {
    ...mapGetters('createDataset', ['getFilesSource', 'getProject']),
    project: {
      set(value) {
        this.setProject(value);
      },
      get() {
        return this.getProject;
      },
    },
  },
  methods: {
    ...mapActions('createDataset', ['getDatasetSources', 'setSelectSource']),
    isActive(tab) {
      return this.project.active === tab;
    },
    onSelect(data) {
      this.setSelectSource(data);
    },
    onTabs(tab) {
      this.project.active = tab;
      this.project.source_path = '';
      this.setSelectSource({});
    },
  },
};
</script>

<style lang="scss" scoped>
@import '@/assets/scss/variables/default.scss';
.state-one {
  &-list {
    height: 50px;
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
          bottom: -5px;
          left: 50%;
          background: $color-light-blue;
          transform: translateX(-50%);
        }
      }
    }
  }
}
</style>
