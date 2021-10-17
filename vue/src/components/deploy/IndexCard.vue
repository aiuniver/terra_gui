<template>
  <div class="card">
    <div class="card__content">
      <div v-if="type == 'image_classification'">
        <div class="card__original">
          <ImgCard :imgUrl="card.source" />
        </div>
        <div class="card__result">
          <TextCard :style="{ width: '224px', height: '80px' }">{{ imageClassificationText }}</TextCard>
        </div>
      </div>
      <div v-if="type == 'text_classification'">
        <div class="card__original">
          <TextCard :style="{ width: '600px', color: '#A7BED3', height: '324px' }">{{ card.source }}</TextCard>
        </div>
        <div v-if="type == 'audio_classification'">
          <div class="card__original">
            <!--        <TextCard :style="{ width: '600px', color: '#A7BED3', height: '324px' }">{{ card.source }}</TextCard>-->
            <AudioCard :value="card.source" :update="RandId" />
          </div>
          <div class="card__result">
            <TextCard :style="{ width: '600px', height: '80px' }">{{ imageClassificationText }}</TextCard>
          </div>
        </div>
      </div>
      <div v-if="type == 'text_segmentation'">
        <div class="card__original">
          <TextCard :style="{ width: '600px', color: '#A7BED3', height: '324px' }">{{ card.format }}</TextCard>
        </div>
        <div class="card__result">
          <TextCard :style="{ width: '600px', height: '80px' }">{{ card.format }}</TextCard>
        </div>
      </div>
      <div v-if="type == 'audio_classification'">
        <div class="card__original">
          <!--        <TextCard :style="{ width: '600px', color: '#A7BED3', height: '324px' }">{{ card.source }}</TextCard>-->
          <AudioCard :value="card.source" :update="RandId" />
        </div>
        <div class="card__result">
          <TextCard :style="{ width: '600px', height: '80px' }">{{ imageClassificationText }}</TextCard>
        </div>
      </div>

      <div v-if="type == 'image_segmentation'">
        <div class="card__original">
          <ImgCard :imgUrl="card.source" />
        </div>
        <div class="card__result">
          <ImgCard :imgUrl="card.segment" />
        </div>
      </div>
      <div class="card__graphic" v-if="type == 'graphic'">
        <Plotly :data="card.data" :layout="layout" :display-mode-bar="false"></Plotly>
      </div>

    </div>
  </div>
</template>

<script>
import ImgCard from './cards/ImgCard';
import TextCard from './cards/TextCard';
import AudioCard from './cards/AudioCard';
import { Plotly } from 'vue-plotly';
import { mapGetters } from 'vuex';
export default {
  name: 'IndexCard',
  components: {
    ImgCard,
    TextCard,
    Plotly,
    AudioCard,
  },
  data: () => ({}),
  props: {
    card: {
      type: Object,
      default: () => ({}),
    },
    index: [String, Number],
  },

  methods: {
    ReloadCard() {
      this.$emit('reload', [this.index.toString()]);
    },
  },
  mounted() {
    console.log(this.card);
    console.log(this.type);
  },
  computed: {
    ...mapGetters({
      graphicData: 'deploy/getGraphicData',
      defaultLayout: 'deploy/getDefaultLayout',
      origTextStyle: 'deploy/getOrigTextStyle',
      type: 'deploy/getDeployType',
      RandId: 'deploy/getRandId',
    }),
    layout() {
      const layout = this.defaultLayout;
      if (this.char) {
        layout.title.text = this.char.title || '';
        layout.xaxis.title = this.char.xaxis.title || '';
        layout.yaxis.title = this.char.yaxis.title || '';
      }
      return layout;
    },
    imageClassificationText() {
      let text = this.card.data;
      let prepareText = '';
      text.sort((a, b) => (a[1] < b[1] ? 1 : -1));
      for (let i = 0; i < text.length; i++) {
        prepareText = prepareText + `${text[i][0]} - ${text[i][1]}% \n`;
      }
      return prepareText;
    },
  },
};
</script>

<style lang="scss" scoped>
.card__reload {
  padding-left: 5px;
}
.card {
  padding: 15px 15px 15px 0;
  display: flex;
}
.card__graphic {
  background: #242f3d;
  border: 1px solid #6c7883;
  box-sizing: border-box;
  border-radius: 4px;
}
.card__original {
  background: #242f3d;
}
.card__result {
  padding-top: 6px;
}
.btn-reload {
  width: 32px;
  height: 32px;
  i {
    position: absolute;
    margin-left: 7px;
    margin-top: -13px;
    width: 16px;
  }
}
.card__table {
  width: 100%;
}
</style>