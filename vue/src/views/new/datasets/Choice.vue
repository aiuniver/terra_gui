<template>
  <div class="page-choice">
    <div class="page-choice__menu">
      <ChoiceMenu @select="selectedType = $event" 
      :selectedType="selectedType" 
      :tags="tags" 
      :selectedTag="selectedTag"
      @tagClick="handleTag"/>
    </div>
    <div class="page-choice__main">
      <Datasets :datasets="filteredList" :selectedType="selectedType" @choice="getVersions" />
    </div>
    <at-modal
      v-model="showModal"
      :showConfirmButton="false"
      :showCancelButton="false"
      :width="450"
      title="Выбор версии"
      @on-cancel="onCancel"
    >
      <div v-for="(item, idx) in versions"
      class="page-choice__versions"
      :key="idx" 
      @click="selectedVersion = item"
      :class="{ 'active': item.alias === selectedVersion.alias }"
      >
        <span class="name">{{ item.name }}</span>
        <span class="info">{{ getSize(item.size) }}</span>
        <span class="info">{{ getDate(item.date) }}</span>
      </div>
      <template v-slot:footer>
        <d-button @click="setChoice" :disabled="!selectedVersion.alias" style="flex-basis: 50%;">Выбрать</d-button>
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
    selectedVersion: {},
    selectedTag: {},
    tags: []
  }),
  computed: {
    filteredList() {
      if (!this.selectedTag.type) return this.datasets
      return this.datasets.filter(item => {
        if (this.selectedTag.type === 'group') return this.selectedTag.name === item.architecture
        return item.tags.includes(this.selectedTag.name) && this.selectedTag.group === item.architecture
      })
    }
  },
  methods: {
    getDate(val) {
      if (!val) return ''
      return new Date(val).toLocaleString('ru-RU', { dateStyle: 'short', timeStyle: 'short' })
    },
    getSize(size) {
      if (size?.value) return `${size.short.toFixed(2)} ${size.unit}`
      return ''
    },
    onCancel() {
      this.versions = []
      this.selectedVersion = {}
    },
    async getVersions({ alias, group }) {
      this.showModal = true
      const { data } = await this.$store.dispatch('axios', { url: '/datasets/versions/', data: { group, alias } })
      this.selectedSet = { group, alias }
      this.versions = data
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
    },
    handleTag(e) {
      if (e.type === this.selectedTag.type && e.name === this.selectedTag.name) return this.selectedTag = {};
      this.selectedTag = e;
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
    this.$store.commit('datasets/SET_GROUPS', data.groups)
    this.tags = data.tags
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
    width: 250px;
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
    cursor: pointer;
    padding: 5px;
    border-bottom: 2px solid #1E2734;
    display: flex;
    gap: 10px;
    align-items: center;
    .name {
      flex-grow: 1;
      word-wrap: anywhere;
      overflow-wrap: anywhere;
    }
    .info {
      font-size: 12px;
      color: #A7BED3;
      white-space: nowrap;
    }
    &:hover, &.active {
      background: #1E2734;
      color: #65B9F4;
    }
  }
}
</style>