<template>
  <div class="state-one">
    <div class="state-one-list flex align-center">
      <div
        v-for="{ text, mode } in items"
        :class="['state-one-list__item', { 'state-one-list__item--active': isActive(mode) }]"
        :key="`tab_${mode}`"
        @click="onTabs(mode)"
      >
        {{ text }}
      </div>
    </div>
    <div class="state-one-content mt-10">
      <t-field v-if="project.source.mode === 'GoogleDrive'" icon="google" label="Выберите файл на Google диске">
        <d-auto-complete
          :value="getValueSource"
          :key="getValueSource"
          placeholder="Введите имя файла"
          :list="getFilesSource"
          @click="getDatasetSources"
          @change="onSelect({ mode: 'GoogleDrive', value: $event.value })"
        />
      </t-field>
      <t-field v-else icon="link" label="Загрузите по ссылке">
        <d-input-text v-model="project.source.value" placeholder="URL" @blur="onSelect({ mode: 'URL', value: $event.target.value })" />
      </t-field>
    </div>
    <div>
      <t-field label="Название датасета">
        <d-input-text v-model="project.name" />
      </t-field>
      <t-field label="Тип архитектуры">
        <!-- <d-auto-complete :value="getValueArchitectures" placeholder="Архитектуры" :list="getArchitectures" @change="onArchitectures" /> -->
        <d-select :value="getValueArchitectures" placeholder="Архитектуры" :list="getArchitectures" @change="onArchitectures" />
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
      { text: 'Google диск', mode: 'GoogleDrive' },
      { text: 'URL', mode: 'URL' },
    ],
  }),
  computed: {
    ...mapGetters('createDataset', ['getFilesSource', 'getProject', 'getArchitectures']),
    project: {
      set(value) {
        this.setProject(value);
      },
      get() {
        return this.getProject;
      },
    },
    getValueSource() {
      const value = this.project.source.value;
      return this.getFilesSource.find(i => i.value === value)?.label || '';
    },
    getValueArchitectures() {
      const value = this.project.architecture;
      return this.getArchitectures.find(i => i.value === value)?.label || '';
    },
  },
  methods: {
    ...mapActions('createDataset', ['getDatasetSources']),
    isActive(mode) {
      return this.project.source.mode === mode;
    },
    onSelect({ value, mode }) {
      this.project.source.mode = mode;
      this.project.source.value = value;
    },
    onTabs(mode) {
      this.project.source.mode = mode;
      this.project.source.value = '';
    },
    onArchitectures({ value }) {
      console.log(value);
      this.project.architecture = value;
    },
  },
  created() {
    this.getDatasetSources();
  },
};
</script>

<style lang="scss">
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
