<template>
  <div :class="['project', {'project--selected': active}]">
    <div class="project-edging">
      <svg width="300" height="196" viewBox="0 0 300 196" fill="none" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern :id="image" x="0" y="0" patternUnits="userSpaceOnUse" height="100%" width="100%">
            <image x="0" y="0" width="100%" height="100%" :xlink:href="image" />
          </pattern>
        </defs>
        <path :fill="`url(#${image})`" d="M167.712 15.5H1.26079L16.1875 1.4513C16.8368 0.84025 17.6947 0.5 18.5863 0.5H149.947C150.821 0.5 151.664 0.827078 152.309 1.4169L167.712 15.5ZM169 16.5H170.288H260.304L280.5 35.2183V165.276C280.5 166.239 280.103 167.159 279.403 167.82L263.813 182.545C263.163 183.158 262.303 183.5 261.41 183.5H122.711C122.02 183.5 121.345 183.295 120.77 182.912L97.285 167.256C96.5458 166.763 95.6773 166.5 94.7889 166.5H26.149L0.5 149.73V16.5H168.806H169Z" stroke="#65B9F4"/>
        <path d="M16.1875 1.4513C16.8368 0.84025 17.6947 0.5 18.5863 0.5H150.181C151.063 0.5 151.913 0.833577 152.561 1.4339L167.726 15.5H1.26079L16.1875 1.4513Z" fill="#65B9F4" stroke="#65B9F4"/>
        <path d="M280.5 19.7214V30.8866L264.315 16.5L277.243 16.5001C279.046 16.5001 280.5 17.9467 280.5 19.7214Z" stroke="#65B9F4"/>
        <path d="M0 152L23 167H4C1.79086 167 0 165.209 0 163V152Z" fill="#65B9F4"/>
      </svg>

      <div class="project-created">
        <span>{{ created }}</span>
      </div>
      <div class="project-edited">
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M2.21 10.2895C2.06974 10.2892 1.93603 10.2301 1.8415 10.1265C1.74522 10.0237 1.69738 9.88471 1.71 9.74445L1.8325 8.39745L7.4915 2.74045L9.26 4.50845L3.6025 10.165L2.2555 10.2875C2.24 10.289 2.2245 10.2895 2.21 10.2895ZM9.613 4.15495L7.845 2.38695L8.9055 1.32645C8.99928 1.23256 9.12654 1.17981 9.25925 1.17981C9.39195 1.17981 9.51921 1.23256 9.613 1.32645L10.6735 2.38695C10.7674 2.48074 10.8201 2.608 10.8201 2.7407C10.8201 2.87341 10.7674 3.00067 10.6735 3.09445L9.6135 4.15445L9.613 4.15495Z" fill="#6C7883"/>
          </svg>

        <span>{{ edited }}</span> 
      </div>
      <div class="project-bar" @click="isSettings = !isSettings">
        <svg width="14" height="4" viewBox="0 0 14 4" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 3.66665C11.0796 3.66665 10.3334 2.92045 10.3334 1.99998C10.3334 1.0795 11.0796 0.333313 12 0.333313C12.9205 0.333313 13.6667 1.0795 13.6667 1.99998C13.6667 2.44201 13.4911 2.86593 13.1786 3.17849C12.866 3.49105 12.4421 3.66665 12 3.66665ZM7.00004 3.66665C6.07957 3.66665 5.33337 2.92045 5.33337 1.99998C5.33337 1.0795 6.07957 0.333313 7.00004 0.333313C7.92051 0.333313 8.66671 1.0795 8.66671 1.99998C8.66671 2.44201 8.49111 2.86593 8.17855 3.17849C7.86599 3.49105 7.44207 3.66665 7.00004 3.66665ZM2.00004 3.66665C1.07957 3.66665 0.333374 2.92045 0.333374 1.99998C0.333374 1.0795 1.07957 0.333313 2.00004 0.333313C2.92052 0.333313 3.66671 1.0795 3.66671 1.99998C3.66671 2.44201 3.49111 2.86593 3.17855 3.17849C2.86599 3.49105 2.44207 3.66665 2.00004 3.66665Z" fill="#242F3D"/>
        </svg>
        <div class="project-menu" v-if="isSettings">
          <p @click="$emit('editProject', id)">
            <svg width="12" height="13" viewBox="0 0 12 13" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M0.762496 12.3618C0.587173 12.3615 0.420039 12.2876 0.301871 12.1581C0.181524 12.0296 0.121724 11.8559 0.137496 11.6806L0.290621 9.99685L7.36437 2.9256L9.575 5.1356L2.50312 12.2062L0.819371 12.3593C0.799996 12.3612 0.780621 12.3618 0.762496 12.3618ZM10.0162 4.69372L7.80625 2.48372L9.13187 1.1581C9.2491 1.04074 9.40818 0.974792 9.57406 0.974792C9.73994 0.974792 9.89902 1.04074 10.0162 1.1581L11.3419 2.48372C11.4592 2.60095 11.5252 2.76003 11.5252 2.92591C11.5252 3.09179 11.4592 3.25087 11.3419 3.3681L10.0169 4.6931L10.0162 4.69372Z"/>
            </svg>

            <span>Редактировать</span>
          </p>
          <p @click="$emit('deleteProject', id)">
            <svg width="13" height="14" viewBox="0 0 13 14" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M9.625 13.25H3.375C2.68464 13.25 2.125 12.6904 2.125 12V3.875H0.875V2.625H3.375V2C3.375 1.30964 3.93464 0.75 4.625 0.75H8.375C9.06536 0.75 9.625 1.30964 9.625 2V2.625H12.125V3.875H10.875V12C10.875 12.6904 10.3154 13.25 9.625 13.25ZM3.375 3.875V12H9.625V3.875H3.375ZM4.625 2V2.625H8.375V2H4.625Z"/>
            </svg>

            <span>Удалить</span>
          </p>
        </div>
      </div>
    </div>

    <!-- <div class="project-image">
      <img :src="image" :alt="headline">
    </div> -->
    <div class="project-headline">
      <h3>{{ headline }}</h3>
    </div>
  </div>
</template>

<script>
export default {
  name: 'CardProject',
  props:['image', 'created', 'edited', 'headline', 'active', 'id'],
  data: () => ({
    isSettings: false
  })
}
</script>

<style lang="scss" scoped>
.project{
  position: relative;

  &-edging{
    position: relative;
  }

  &-bar{
    position: absolute;
    top: -6px;
    left: 25px;
    cursor: pointer;
  }

  &-menu{
    background: #242F3D;
    display: flex;
    flex-direction: column;
    border-radius: 4px;
    overflow: hidden;
    position: relative;
    top: 5px;
    p{
      padding: 8px 12px;
      display: flex;
      align-items: center;

      &:hover{
        background: #0E1621;
        span{
          color: #65B9F4;
        }
        svg{
          fill: #65B9F4;
        }
      }

      span{
        font-family: "Open Sans";
        font-size: 14px;
        color: #A7BED3;
        margin-left: 8px;
      }
    }
    svg{
      fill: #A7BED3;
    }
  }

  &-image{
    background: #111111;
    border-radius: 4px;
    height: 170px;
    position: absolute;
    top:0;

    img{
      width: 100%;
      object-fit: cover;
      height: 100%;
    }
  }


  &-headline{
    margin-top: -10px;
    h3{
      font-family: "Open Sans", sans-serif;
      font-weight: normal;
      font-size: 14px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
  }

  &-created{
    position: absolute;
    bottom: 12px;
    span{
      color: #6C7883;
      font-size: 11px;
    }
  }

  &-edited{
    position: absolute;
    top: -4px;
    right: 45px;
    display: flex;
    align-items: center;

    span{
      color: #6C7883;
      font-size: 11px;
      margin-left: 5px;
    }
  }
}
</style>