<template>
  <div class="page-choice">
    <div class="page-choice__menu">
      <ChoiceMenu @select="selectedType = $event" :selectedType="selectedType"/>
    </div>
    <div class="page-choice__main">
      <Datasets :datasets="datasets" :selectedType="selectedType" @choice="getVersions" />
    </div>
    <at-modal
      v-model="showModal"
      :showConfirmButton="false"
      :showCancelButton="false"
      :width="400"
      title="Выбор версии"
      @on-cancel="onCancel"
    >
      <ul class="page-choice__versions">
        <li v-for="(item, idx) in versions" 
        :key="idx" 
        @click="selectedVersion = item"
        :class="{ 'active': item.value === selectedVersion.value }"
        >{{ item.label }}</li>
      </ul>
      <template v-slot:footer>
        <d-button @click="setChoice" :disabled="!selectedVersion.value" style="flex-basis: 50%;">Выбрать</d-button>
      </template>
    </at-modal>
  </div>
</template>

<script>
import Datasets from '@/components/datasets/choice/';
import ChoiceMenu from '@/components/datasets/choice/ChoiceMenu';

export default {
  name: 'Choice',
  components: {
    Datasets,
    ChoiceMenu
  },
  data: () => ({
    selectedType: 2,
    datasets: [],
    showModal: false,
    versions: [],
    tID: null,
    selectedSet: {},
    selectedVersion: {}
  }),
  methods: {
    onCancel() {
      this.versions = []
      this.selectedVersion = {}
    },
    async getVersions({ alias, group }) {
      this.showModal = true
      const { data } = await this.$store.dispatch('axios', { url: '/datasets/versions/', data: { group, alias } })
      this.selectedSet = { group, alias }
      this.versions = data.map(item => ({
        label: item.name,
        value: item.alias
      }))
    },
    async setChoice() {
      this.$store.dispatch('settings/setOverlay', true);
      const { group, alias } = this.selectedSet
      const { success } = await this.$store.dispatch('datasets/choice', { group, alias, version: this.selectedVersion.value })

      if (success) {
        this.$store.dispatch('messages/setMessage', { message: `Загружаю датасет` });
        this.createInterval();
      }
    },
    createInterval() {
      this.tID = setTimeout(async () => {
        const res = await this.$store.dispatch('datasets/choiceProgress', {});
        if (res) {
          const { data } = res;
          if (data) {
            const { finished, message, percent, error } = data;
            this.$store.dispatch('messages/setProgressMessage', message);
            this.$store.dispatch('messages/setProgress', percent);
            if (finished) {
              this.$store.dispatch('messages/setProgress', 0);
              this.$store.dispatch('messages/setProgressMessage', '');
              await this.$store.dispatch('projects/get');
              this.$store.dispatch('messages/setMessage', {
                message: `Датасет «${data?.data?.dataset?.name || ''}» выбран`,
              });
              this.$store.dispatch('settings/setOverlay', false);
              this.showModal = false
              this.onCancel()
            } else {
              if (error) {
                // this.$store.dispatch('messages/setMessage', { error });
                this.$store.dispatch('messages/setProgressMessage', '');
                this.$store.dispatch('messages/setProgress', 0);
                this.$store.dispatch('settings/setOverlay', false);
                return;
              }
              this.createInterval();
            }
          } else {
            this.$store.dispatch('settings/setOverlay', false);
          }
        } else {
          this.$store.dispatch('settings/setOverlay', false);
        }
      }, 1000);
    }
  },
  async created() {
    const { data } = await this.$store.dispatch('axios', { url: '/datasets/info/' })
    data.datasets.forEach(item => {
      item.datasets.forEach(dataset => {
        this.datasets.push({
          ...dataset,
          group: item.alias
        })
      })
    })
  }
};
</script>

<style lang="scss">
@import "@/assets/scss/variables/default.scss";
.page-choice {
  position: relative;
  display: flex;
  height: 100%;
  &__menu {
    width: 175px;
    border-right: 1px solid $color-black;
    padding-top: 18px;
    hr {
      border: none;
      border-bottom: 1px solid $color-dark-gray;
      margin: 20px;
    }
  }
  &__main {
    width: 100%;
  }
  &__versions {
    font-size: 14px;
    max-height: 400px;
    overflow-y: auto;
    li {
      cursor: pointer;
      padding: 5px;
      border-bottom: 2px solid #1E2734;
      &:hover, &.active {
        background: #1E2734;
        color: #65B9F4;
      }
    }
  }
}
</style>