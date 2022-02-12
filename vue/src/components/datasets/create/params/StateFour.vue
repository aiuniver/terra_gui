<template>
  <div class="state-four">
    <div>
      <t-field label="Название версии">
        <d-input-text v-model="project.verName" />
      </t-field>
      <div>
        <DSlider v-model="project.train" />
      </div>
      <t-field label="Перемешать">
        <d-checkbox v-model="project.shuffle"></d-checkbox>
      </t-field>
    </div>
  </div>
</template>

<script>
import { mapActions, mapGetters } from 'vuex';
import DSlider from '@/components/forms/DSlider';
export default {
  name: 'state-four',
  components: {
    DSlider,
  },
  data: () => ({
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
  },
};
</script>

<style lang="scss" scoped>
.state-four {
  &-list {
    &__menu {
      max-width: 320px;
      li {
        margin-bottom: 25px;
        span {
          margin-right: auto;
        }
      }
    }
  }
}
</style>